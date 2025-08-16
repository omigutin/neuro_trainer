# src/neuro_trainer/app_facade.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Any

from .state import AppState, state_to_dict, state_from_dict
from .config_store import ConfigStore
from .models_registry import ModelsRegistry
from .presets import PresetRegistry, Preset
from .trainer import Trainer
from .metrics import MetricsStore  # noqa: F401  # (используется внутри Trainer)
from .utils import AppConfigPaths

from .services.validator import ParamValidator
from .services.task_resolver import TaskResolver
from .services.imgsz import normalize_imgsz
from .services.run_namer import resolve_run_name
from .services.weights_resolver import WeightsResolver
from .weights_cache import WeightsCache


class NeuroTrainerApp:
    """
    Фасад над сервисами — orchestration.
    """

    def __init__(self, paths: AppConfigPaths, config_store: ConfigStore,
                 models_registry: ModelsRegistry, presets: PresetRegistry) -> None:
        self._paths = paths
        self._config_store = config_store
        self._models = models_registry
        self._presets = presets

        self._state: AppState = AppState(project_dir=str(paths.default_runs_dir.as_posix()))
        self._trainer: Optional[Trainer] = None

        # сервисы
        self._weights = WeightsResolver(WeightsCache(paths.weights_dir))

    # ---------- I/O конфигов ----------

    def load_last_session(self) -> None:
        cfg = self._config_store.load_last_session()
        if cfg:
            self._state = state_from_dict(cfg, self._paths.default_runs_dir.as_posix())

    def save_last_session(self) -> None:
        self._config_store.save_last_session(state_to_dict(self._state))

    def load_preset_yaml(self, path: Path) -> None:
        cfg = self._config_store.load_yaml(path)
        self._state = state_from_dict(cfg, self._paths.default_runs_dir.as_posix())

    def save_preset_yaml(self, path: Path) -> None:
        self._config_store.save_yaml(path, state_to_dict(self._state))

    # ---------- выбор модели/профиля ----------

    def set_model_by_name(self, name: str) -> None:
        self._require(self._models.is_known(name), f"Неизвестная модель: {name}")
        self._state.model_name = name
        self._state.model_path = ""
        self._state.task_override = ""

    def set_model_by_path(self, path: Path) -> None:
        p = path.resolve()
        self._require(p.exists() and p.is_file(), f"Файл не найден: {p}")
        self._state.model_path = str(p)
        self._state.model_name = ""

    def set_task_override(self, task: Optional[str]) -> None:
        if task is None:
            self._state.task_override = ""
            return
        self._require(task in ("classifier", "detector", "segmentation"), "Некорректный тип задачи")
        self._state.task_override = str(task)

    def set_profile(self, profile: str) -> None:
        self._require(profile in ("fast", "quality", "low_vram"), "Профиль должен быть fast/quality/low_vram")
        self._state.profile = str(profile)

    # ---------- параметры ----------

    def set_params(self, **kwargs: Any) -> None:
        s = self._state
        if "data" in kwargs and kwargs["data"] is not None:
            s.data = str(kwargs["data"]).strip()

        if "project_dir" in kwargs and kwargs["project_dir"] is not None:
            pd = Path(kwargs["project_dir"]).resolve()
            pd.mkdir(parents=True, exist_ok=True)
            s.project_dir = str(pd)

        if "name" in kwargs and kwargs["name"] is not None:
            s.name = str(kwargs["name"]).strip()

        if "auto_increment_name" in kwargs and kwargs["auto_increment_name"] is not None:
            s.auto_increment_name = bool(kwargs["auto_increment_name"])

        if "epochs" in kwargs and kwargs["epochs"] is not None:
            s.epochs = ParamValidator.range_int("epochs", int(kwargs["epochs"]), 1, 1000)

        if "patience" in kwargs and kwargs["patience"] is not None:
            s.patience = ParamValidator.range_int("patience", int(kwargs["patience"]), 0, 300)

        if "batch" in kwargs and kwargs["batch"] is not None:
            s.batch = ParamValidator.range_int("batch", int(kwargs["batch"]), 1, 1024)

        if "imgsz" in kwargs and kwargs["imgsz"] is not None:
            imgsz = kwargs["imgsz"]
            if isinstance(imgsz, tuple):
                ParamValidator.require(len(imgsz) == 2 and all(isinstance(x, int) for x in imgsz),
                                       "imgsz должен быть int или (H, W)")
                ParamValidator.require(all(64 <= x <= 2048 for x in imgsz),
                                       "каждое измерение imgsz должно быть в [64..2048]")
                s.imgsz = (int(imgsz[0]), int(imgsz[1]))
            else:
                ParamValidator.require(isinstance(imgsz, int), "imgsz должен быть int или (H, W)")
                ParamValidator.require(64 <= imgsz <= 2048, "imgsz должен быть в [64..2048]")
                s.imgsz = int(imgsz)

        if "rect" in kwargs and kwargs["rect"] is not None:
            s.rect = bool(kwargs["rect"])

        if "multi_scale" in kwargs and kwargs["multi_scale"] is not None:
            s.multi_scale = bool(kwargs["multi_scale"])

        if "seed" in kwargs and kwargs["seed"] is not None:
            s.seed = int(kwargs["seed"])

        if "deterministic" in kwargs and kwargs["deterministic"] is not None:
            s.deterministic = bool(kwargs["deterministic"])

        if "workers" in kwargs and kwargs["workers"] is not None:
            s.workers = ParamValidator.range_int("workers", int(kwargs["workers"]), 0, 64)

        if "device" in kwargs and kwargs["device"] is not None:
            ParamValidator.require(kwargs["device"] in ("auto", "cuda", "mps", "cpu"),
                                   "device должен быть auto/cuda/mps/cpu")
            s.device = str(kwargs["device"])

    # ---------- валидация/жизненный цикл ----------

    def validate(self) -> tuple[bool, str]:
        res = ParamValidator.basic(self._state.model_name, self._state.model_path, self._state.data)
        return (res.ok, res.message)

    def run(self) -> None:
        self.load_last_session()

    def validate_only(self) -> None:
        ok, msg = self.validate()
        self._require(ok, msg)
        tr = self._prepare_trainer()
        tr.validate_only()
        self._trainer = tr
        self.save_last_session()

    def train(self) -> None:
        ok, msg = self.validate()
        self._require(ok, msg)
        tr = self._prepare_trainer()
        tr.train()
        self._trainer = tr
        self.save_last_session()

    # ---------- utils для GUI ----------

    def open_last_run_folder(self) -> Optional[Path]:
        if self._trainer is None:
            return None
        return Path(self._trainer.save_dir)

    def open_last_log(self) -> Optional[Path]:
        if self._trainer is None:
            return None
        return self._trainer.jsonl_log_path

    # ---------- internal ----------

    def _prepare_trainer(self) -> Trainer:
        s = self._state

        # task
        task = TaskResolver.resolve(self._models, s.model_name, s.task_override)

        # weights (локальный путь)
        weights = self._weights.resolve(s.model_name, s.model_path)

        # preset
        preset_name = self._presets.resolve_name(task, s.profile)
        preset: Preset = self._presets.get(preset_name)

        # imgsz
        imgsz = normalize_imgsz(s.imgsz, task)

        # project/name
        project_dir = Path(s.project_dir).resolve()
        run_name = resolve_run_name(project_dir, task, s.name, s.auto_increment_name)

        return Trainer(
            weights=weights,
            task_hint=task,
            data=s.data,
            project_dir=str(project_dir),
            run_name=run_name,
            epochs=s.epochs,
            patience=s.patience,
            batch=s.batch,
            imgsz=imgsz,
            rect=s.rect,
            multi_scale=s.multi_scale,
            seed=s.seed,
            deterministic=s.deterministic,
            workers=s.workers,
            device=s.device,
            preset=preset,
        )

    @staticmethod
    def _require(cond: bool, msg: str) -> None:
        if not cond:
            raise ValueError(msg)
