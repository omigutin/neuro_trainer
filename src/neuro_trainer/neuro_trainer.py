from __future__ import annotations
"""
NeuroTrainerApp — фасад: состояние настроек, профили, пользовательские пресеты
и сборка задания на обучение (JobSpec) для запуска в отдельном процессе.
"""

from pathlib import Path
from typing import Any

from .config_store import ConfigStore
from .job import JobKind, JobSpec
from .model_inspector import WeightsInfo, inspect_weights
from .models_registry import ModelsRegistry
from .params_catalog import coerce_values, has_param, train_kwargs
from .presets import PresetRegistry
from .services.imgsz import normalize_imgsz
from .services.run_namer import resolve_run_name
from .services.validator import ParamValidator
from .state import AppState, ModelSource, Profile, TrainMode, state_from_dict, state_to_dict
from .tasks import TaskKind, parse_task
from .utils import AppConfigPaths


class NeuroTrainerApp:
    """ Фасад над сервисами — orchestration. """

    def __init__(self, paths: AppConfigPaths, config_store: ConfigStore, presets: PresetRegistry) -> None:
        self._paths = paths
        self._config_store = config_store
        self._presets = presets
        self._state = AppState(project_dir=str(paths.default_runs_dir))

    # ---------- состояние ----------

    @property
    def paths(self) -> AppConfigPaths:
        return self._paths

    def state_dict(self) -> dict[str, Any]:
        """ Текущее состояние в плоском виде (для формы). """
        return state_to_dict(self._state)

    def update(self, values: dict[str, Any]) -> dict[str, str]:
        """
        Применить значения из формы. Корректные значения сохраняются в состоянии,
        для ошибочных возвращается {ключ: сообщение}.
        """
        errors: dict[str, str] = {}
        s = self._state
        try:
            if "model_source" in values:
                s.model_source = ModelSource(values["model_source"])
            if "train_mode" in values:
                s.train_mode = TrainMode(values["train_mode"])
            if "profile" in values:
                s.profile = Profile(values["profile"])
            if "task" in values:
                s.task = parse_task(values["task"])
        except ValueError as e:
            errors["model"] = str(e)
        for key in ("model_name", "model_path", "data", "project_dir", "name"):
            if key in values:
                setattr(s, key, str(values[key] or "").strip())
        if "auto_increment_name" in values:
            s.auto_increment_name = bool(values["auto_increment_name"])
        if s.model_source is ModelSource.STANDARD and s.model_name:
            if ModelsRegistry.is_known(s.model_name):
                s.task = ModelsRegistry.infer_task_from_name(s.model_name)
            else:
                errors["model_name"] = f"Unknown standard model: {s.model_name}"

        params, param_errors = coerce_values({k: v for k, v in values.items() if has_param(k)})
        s.params.update(params)
        errors.update(param_errors)
        return errors

    def load_last_session(self) -> None:
        cfg = self._config_store.load_last_session()
        if cfg:
            self._state = state_from_dict(cfg, str(self._paths.default_runs_dir))

    def save_last_session(self) -> None:
        self._config_store.save_last_session(state_to_dict(self._state))

    # ---------- профили и пользовательские пресеты ----------

    def profile_values(self, task: TaskKind, profile: Profile) -> dict[str, Any]:
        """ Гиперпараметры профиля для задачи (без применения к состоянию). """
        return self._presets.get(task, profile)

    def list_user_presets(self) -> list[str]:
        return self._config_store.list_presets()

    def save_user_preset(self, name: str) -> None:
        self._config_store.save_preset(_safe_name(name), state_to_dict(self._state))

    def load_user_preset(self, name: str) -> None:
        data = self._config_store.load_preset(_safe_name(name))
        if not data:
            raise FileNotFoundError(f"Preset '{name}' not found")
        self._state = state_from_dict(data, str(self._paths.default_runs_dir))

    # ---------- своя модель ----------

    def inspect_custom_model(self, path: str) -> WeightsInfo:
        """ Прочитать сведения о своих весах и выставить задачу из файла. """
        info = inspect_weights(path)
        self._state.model_source = ModelSource.CUSTOM
        self._state.model_path = info.path
        self._state.task = info.task
        return info

    # ---------- задание ----------

    def build_job(self, kind: JobKind) -> JobSpec:
        """ Проверить состояние и собрать задание для процесса обучения. """
        s = self._state
        custom = s.model_source is ModelSource.CUSTOM
        resume = custom and s.train_mode is TrainMode.RESUME and kind is JobKind.TRAIN
        res = ParamValidator.basic("" if custom else s.model_name, s.model_path if custom else "", s.data,
                                   need_data=not resume)
        ParamValidator.require(res.ok, res.message)
        if custom:
            ParamValidator.require(Path(s.model_path).is_file(), f"Файл весов не найден: {s.model_path}")
        if resume:
            ParamValidator.require(Path(s.model_path).name == "last.pt",
                                   "Для продолжения прерванного обучения выберите weights/last.pt этого рана.")

        params = dict(s.params)
        params["imgsz"] = normalize_imgsz(params["imgsz"], s.task)
        project_dir = Path(s.project_dir or self._paths.default_runs_dir).resolve()
        run_name = "" if resume else resolve_run_name(project_dir, s.task, s.name, s.auto_increment_name)
        return JobSpec(
            kind=kind,
            task=s.task,
            data=_absolute_if_exists(s.data),
            project_dir=str(project_dir),
            run_name=run_name,
            model_name="" if custom else s.model_name,
            model_path=str(Path(s.model_path).resolve()) if custom else "",
            weights_dir=str(self._paths.weights_dir.resolve()),
            train_args=train_kwargs(params, s.task),
            resume=resume,
        )


def _absolute_if_exists(value: str) -> str:
    """ Локальный путь → абсолютный (процесс обучения работает в другом каталоге); URL/имя — как есть. """
    p = Path(value)
    return str(p.resolve()) if value and p.exists() else value


def _safe_name(name: str) -> str:
    """ Имя пользовательского пресета без путей. """
    n = (name or "").strip()
    if not n or any(c in n for c in '\\/:*?"<>|') or n in (".", ".."):
        raise ValueError(f"Invalid preset name: {name!r}")
    return n
