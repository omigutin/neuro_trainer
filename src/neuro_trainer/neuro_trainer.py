from __future__ import annotations

"""
    NeuroTrainerApp — фасад приложения для обучения моделей Ultralytics YOLO (CLS/DET/SEG).
    
    Идея простая: дать «человеческий» интерфейс поверх Ultralytics:
    - выбрать модель (из списка стандартных весов или локальный .pt);
    - автоматически понять тип задачи (классификация / детекция / сегментация);
    - выбрать профиль (fast / quality / low_vram) для выбранного типа;
    - задать базовые параметры (epochs, patience, batch, imgsz) и безопасно их проверить;
    - аккуратно привести imgsz к корректным значениям (кратно 32; для классификации — квадрат);
    - запустить валидацию (быстрый чек путей/датасета/метрик);
    - запустить обучение и сохранить метрики (JSON/CSV рядом с results.csv);
    - хранить/грузить YAML вашей конфигурации, и автоматически сохранять last_session.yaml
      в каталоге настроек (там же, где будут «Загрузить пресет» / «Сохранить пресет»).
    
    Пояснения «на пальцах»:
    - Тип задачи определяется из самих весов модели (у Ultralytics внутри веса знают: это cls/det/seg).
      Если модель нестандартная и тип не распознан — его можно указать вручную.
    - Профиль:
        fast      — «быстро проверить гипотезу» (меньше эпох, мягкие аугментации)
        quality   — «качество важнее скорости» (больше эпох/мешалок, дольше)
        low_vram  — «для слабых видеокарт» (меньше batch и imgsz, чтобы не падать по памяти)
    - imgsz:
        * Классификация — одно число (делаем квадрат). 224 → быстро, 320 → баланс, 384 → качество.
        * Детекция/Сегментация — одно число или пара H×W. Мы доведём до кратности 32.
    
    Зависимости (сервисы), которые внедряются извне (интерфейсы будут реализованы в отдельных файлах):
    - ConfigStore:    загрузка/сохранение YAML (в т.ч. last_session.yaml)
    - ModelsRegistry: список известных моделей (YOLO v8..v11) + определение task по имени/весу
    - PresetRegistry: пресеты параметров: cls/det/seg × fast/quality/low_vram
    - Trainer:        тонкая обёртка над Ultralytics YOLO: train/val, сохранение метрик
    - MetricsStore:   сохранение итоговых метрик (JSON/CSV) рядом с results.csv
    
    GUI (позже PyQt) будет просто вызывать публичные методы NeuroTrainerApp:
    - set_model_by_name(...) / set_model_by_path(...)
    - set_profile(...)
    - set_params(...)
    - validate()
    - train()
    - open_last_run_folder() / open_last_log()
    
    Все методы тщательно валидируют вход, чтобы кнопка «Старт» была недоступна,
    пока значения не «адекватные» (числовые диапазоны, корректный imgsz и т.д.).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Tuple

# Эти классы появятся в соседних модулях пакета (мы их реализуем дальше)
# Здесь — только типы/интерфейс взаимодействия.
from .config_store import ConfigStore  # load_yaml/save_yaml/last_session
from .models_registry import ModelsRegistry  # list_all(), infer_task(name|path)
from .presets import PresetRegistry, Preset  # resolve_name(task, profile), get(name)
from .trainer import Trainer  # train()/validate_only(), свойства save_dir / task / etc.
from .metrics import MetricsStore  # save_json_csv(...)
from .utils import AppConfigPaths  # project_root/configs_dir/default_runs_dir


TaskType = Literal["classifier", "detector", "segmentator"]
ProfileType = Literal["fast", "quality", "low_vram"]


@dataclass(slots=True)
class AppState:
    """Текущее состояние выбранных настроек (то, что обычно редактируется в GUI)."""
    # модель: либо имя из списка, либо локальный .pt
    model_name: str = ""
    model_path: str = ""  # абсолютный путь к .pt, если выбран файл
    # тип задачи (если пусто — попытаемся определить из весов/имени)
    task_override: str = ""
    # профиль fast/quality/low_vram
    profile: ProfileType = "fast"
    # пути
    data: str = ""           # датасет/roboflow/yaml
    project_dir: str = ""    # корень результатов
    name: str = ""           # имя эксперимента (если пусто — авто runN)
    auto_increment_name: bool = True
    # гиперы
    epochs: int = 50
    patience: int = 50
    batch: int = 16
    imgsz: int | tuple[int, int] = 384
    # флаги
    rect: bool = False
    multi_scale: bool = False
    # воспроизводимость/скорость
    seed: int = 0
    deterministic: bool = True
    workers: int = 4
    device: str = "auto"  # auto|cuda|mps|cpu


class NeuroTrainerApp:
    """
    Фасад приложения. Хранит ссылки на сервисы и текущее состояние, валидирует ввод,
    нормализует параметры и управляет жизненным циклом тренировки/валидации.

    Жизненный цикл:
    1) создать экземпляр (внедрить paths, config_store, models_registry, presets);
    2) load_last_session() — подгрузить значения предыдущего запуска (если есть);
    3) пользователь через GUI вызывает set_model_*/set_profile/set_params;
    4) validate() — проверка готовности; если всё ок — train() или validate_only().

    Важные гарантии:
    - imgsz всегда приводится к корректному формату (кратно 32; CLS → int-квадрат).
    - числовые поля всегда в разумных диапазонах.
    - имя эксперимента автогенерируется «runN», если пусто (и инкрементируется при совпадении).
    """

    def __init__(
        self,
        paths: AppConfigPaths,
        config_store: ConfigStore,
        models_registry: ModelsRegistry,
        presets: PresetRegistry,
    ) -> None:
        self._paths = paths
        self._config_store = config_store
        self._models = models_registry
        self._presets = presets

        self._state: AppState = AppState(
            project_dir=str(paths.default_runs_dir.as_posix()),
        )

        # Будет создан при train()/validate_only()
        self._trainer: Optional[Trainer] = None

    # -----------------------
    # Ввод/вывод конфигураций
    # -----------------------

    def load_last_session(self) -> None:
        """Грузит last_session.yaml из каталога конфигов, если есть."""
        cfg = self._config_store.load_last_session()
        if cfg:
            self._state = self._state_from_dict(cfg)

    def save_last_session(self) -> None:
        """Сохраняет текущие настройки в last_session.yaml (рядом с другими YAML)."""
        self._config_store.save_last_session(self._state_to_dict())

    def load_preset_yaml(self, path: Path) -> None:
        """Загружает пользовательский YAML и подменяет текущее состояние."""
        cfg = self._config_store.load_yaml(path)
        self._state = self._state_from_dict(cfg)

    def save_preset_yaml(self, path: Path) -> None:
        """Сохраняет текущие настройки в пользовательский YAML."""
        self._config_store.save_yaml(path, self._state_to_dict())

    # -----------------------
    # Выбор модели и профиля
    # -----------------------

    def set_model_by_name(self, name: str) -> None:
        """
        Задаёт стандартную модель из реестра (YOLO v8..v11, cls/det/seg).
        Тип задачи подставляется автоматически. model_path очищается.
        """
        self._require(self._models.is_known(name), f"Неизвестная модель: {name}")
        self._state.model_name = name
        self._state.model_path = ""
        # если тип известен из имени — оставим task_override пустым (авто)
        self._state.task_override = ""

    def set_model_by_path(self, path: Path) -> None:
        """
        Устанавливает локальный .pt. Тип задачи попытаемся определить из весов во время подготовки Trainer.
        """
        p = path.resolve()
        self._require(p.exists() and p.is_file(), f"Файл не найден: {p}")
        self._state.model_path = str(p)
        self._state.model_name = ""
        # пользователь может потом вручную вызвать set_task_override(), если тип не распознается

    def set_task_override(self, task: TaskType | str | None) -> None:
        """
        Позволяет вручную указать тип задачи (classifier|detector|segmentator) для нестандартных весов.
        """
        if task is None:
            self._state.task_override = ""
            return
        self._require(task in ("classifier", "detector", "segmentator"), "Некорректный тип задачи")
        self._state.task_override = str(task)

    def set_profile(self, profile: ProfileType | str) -> None:
        """Устанавливает профиль fast/quality/low_vram (тип задачи будет стыковаться при подготовке пресета)."""
        self._require(profile in ("fast", "quality", "low_vram"), "Профиль должен быть fast/quality/low_vram")
        self._state.profile = str(profile)  # тип преобразуем позже в имя кодового пресета

    # -----------------------
    # Параметры обучения
    # -----------------------

    def set_params(
        self,
        *,
        data: Optional[str] = None,
        project_dir: Optional[Path] = None,
        name: Optional[str] = None,
        auto_increment_name: Optional[bool] = None,
        epochs: Optional[int] = None,
        patience: Optional[int] = None,
        batch: Optional[int] = None,
        imgsz: Optional[int | tuple[int, int]] = None,
        rect: Optional[bool] = None,
        multi_scale: Optional[bool] = None,
        seed: Optional[int] = None,
        deterministic: Optional[bool] = None,
        workers: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Обновляет выбранные параметры с валидацией «как в GUI»:
        - числовые поля — в разумных диапазонах;
        - imgsz — доводим до корректного формата позже (при подготовке тренера), но уже тут проверяем тип.
        """
        if data is not None:
            self._require(len(data) > 0, "data не может быть пустым")
            self._state.data = data

        if project_dir is not None:
            pd = project_dir.resolve()
            pd.mkdir(parents=True, exist_ok=True)
            self._state.project_dir = str(pd)

        if name is not None:
            self._state.name = name.strip()

        if auto_increment_name is not None:
            self._state.auto_increment_name = bool(auto_increment_name)

        if epochs is not None:
            self._require(1 <= epochs <= 1000, "epochs должно быть в диапазоне [1..1000]")
            self._state.epochs = int(epochs)

        if patience is not None:
            self._require(0 <= patience <= 300, "patience должно быть в диапазоне [0..300]")
            self._state.patience = int(patience)

        if batch is not None:
            self._require(1 <= batch <= 1024, "batch должно быть в диапазоне [1..1024]")
            self._state.batch = int(batch)

        if imgsz is not None:
            if isinstance(imgsz, tuple):
                self._require(len(imgsz) == 2 and all(isinstance(x, int) for x in imgsz), "imgsz должен быть int или (H, W)")
                self._require(all(64 <= x <= 2048 for x in imgsz), "каждое измерение imgsz должно быть в [64..2048]")
                self._state.imgsz = (int(imgsz[0]), int(imgsz[1]))
            else:
                self._require(isinstance(imgsz, int), "imgsz должен быть int или (H, W)")
                self._require(64 <= imgsz <= 2048, "imgsz должен быть в [64..2048]")
                self._state.imgsz = int(imgsz)

        if rect is not None:
            self._state.rect = bool(rect)
        if multi_scale is not None:
            self._state.multi_scale = bool(multi_scale)

        if seed is not None:
            self._state.seed = int(seed)
        if deterministic is not None:
            self._state.deterministic = bool(deterministic)

        if workers is not None:
            self._require(0 <= workers <= 64, "workers должен быть в [0..64]")
            self._state.workers = int(workers)

        if device is not None:
            self._require(device in ("auto", "cuda", "mps", "cpu"), "device должен быть auto/cuda/mps/cpu")
            self._state.device = device

    # -----------------------
    # Подготовка и валидация
    # -----------------------

    def validate(self) -> tuple[bool, str]:
        """
        «Сухая проверка» перед запуском: проверяем, что всё можно стартовать.
        Возвращает (ok, message). Если ok=False — message содержит причину.
        """
        if not (self._state.model_name or self._state.model_path):
            return False, "Не выбрана модель: укажите стандартную или путь к .pt"

        if not self._state.data:
            return False, "Не указан путь к данным (data)"

        # Если выбрана детекция/сегментация — ок; для классификации rect/multi_scale игнорируются
        # Числовые поля уже проверены в set_params(); imgsz «доприведём» в _prepare_trainer()

        return True, "OK"

    # -----------------------
    # Действия
    # -----------------------

    def run(self) -> None:
        """
        Пока GUI нет — просто «инициализируем» приложение.
        Здесь можно подгрузить last_session.yaml и вывести список доступных моделей.
        """
        self.load_last_session()
        # здесь может быть первичная инициализация GUI-модели (когда подключим PyQt)

    def validate_only(self) -> None:
        """Быстрая валидация датасета/модели без обучения — вызовется Trainer.validate_only()."""
        ok, msg = self.validate()
        self._require(ok, msg)

        trainer = self._prepare_trainer()
        trainer.validate_only()
        self._trainer = trainer
        self.save_last_session()

    def train(self) -> None:
        """Запускает обучение с текущими настройками."""
        ok, msg = self.validate()
        self._require(ok, msg)

        trainer = self._prepare_trainer()
        trainer.train()  # внутри сохранит метрики и т.д.
        self._trainer = trainer
        self.save_last_session()

    # -----------------------
    # Утилиты для GUI
    # -----------------------

    def open_last_run_folder(self) -> Optional[Path]:
        """Возвращает путь к последней папке эксперимента (если тренировка уже шла). GUI сам откроет её в проводнике."""
        if self._trainer is None:
            return None
        return Path(self._trainer.save_dir)

    def open_last_log(self) -> Optional[Path]:
        """Если включён JSONL-лог — вернёт путь к лог-файлу (если тренировка шла)."""
        if self._trainer is None:
            return None
        return self._trainer.jsonl_log_path

    # -----------------------
    # Внутренняя кухня
    # -----------------------

    def _prepare_trainer(self) -> Trainer:
        """
        Собирает Trainer из текущего состояния:
        - определяем тип задачи (из имени модели или .pt; при необходимости используем task_override);
        - приводим imgsz к корректному формату (кратно 32; CLS → int-квадрат);
        - подбираем кодовый пресет из PresetRegistry по (task × profile);
        - вычисляем финальное имя рана (runN), если пусто или занято (auto_increment_name=True).
        """
        # 1) Определяем вес и task
        model_name = self._state.model_name.strip()
        model_path = self._state.model_path.strip()
        if model_name:
            self._require(self._models.is_known(model_name), f"Неизвестная модель: {model_name}")
            task = self._models.infer_task_from_name(model_name)  # classifier/detector/segmentator
            weights = model_name
        else:
            self._require(len(model_path) > 0, "Не указан путь к .pt")
            weights = model_path
            # из файла иногда невозможно мгновенно определить тип; Trainer сделает попытку сам,
            # а мы применим task_override, если пользователь задал
            task = self._state.task_override or ""

        # 2) Пресет по типу и профилю
        # если тип пока пуст — Trainer позже уточнит; на этом этапе подберём «разумный» дефолт (detector)
        task_for_preset = task or "detector"
        preset_name = self._presets.resolve_name(task_for_preset, self._state.profile)
        preset: Preset = self._presets.get(preset_name)

        # 3) Нормализуем imgsz (округление до кратного 32; CLS → квадрат)
        imgsz = self._normalize_imgsz(self._state.imgsz, task or task_for_preset)

        # 4) Имя рана (auto runN + инкремент при совпадении)
        project_dir = Path(self._state.project_dir).resolve()
        final_name = self._resolve_run_name(project_dir, (task or task_for_preset), self._state.name, self._state.auto_increment_name)

        # 5) Собираем Trainer
        trainer = Trainer(
            weights=weights,
            task_hint=task,  # может быть пустым — Trainer уточнит из модели
            data=self._state.data,
            project_dir=str(project_dir),
            run_name=final_name,
            # гиперы
            epochs=self._state.epochs,
            patience=self._state.patience,
            batch=self._state.batch,
            imgsz=imgsz,
            rect=self._state.rect,
            multi_scale=self._state.multi_scale,
            seed=self._state.seed,
            deterministic=self._state.deterministic,
            workers=self._state.workers,
            device=self._state.device,
            preset=preset,
        )
        return trainer

    # --- helpers ---

    @staticmethod
    def _require(cond: bool, msg: str) -> None:
        if not cond:
            raise ValueError(msg)

    @staticmethod
    def _to_stride_multiple(v: int, stride: int = 32) -> int:
        """Округляет число до ближайшего кратного stride (обычно 32)."""
        # ближайшее (как пользователь «вышел из поля»)
        return int(round(v / stride) * stride)

    def _normalize_imgsz(self, imgsz: int | tuple[int, int], task: str) -> int | tuple[int, int]:
        """
        Классификация → только int (квадрат), кратный 32.
        Детекция/Сегментация → int или (H, W), каждое кратно 32.
        """
        if task == "classifier":
            if isinstance(imgsz, tuple):
                imgsz = max(imgsz)
            return self._to_stride_multiple(int(imgsz))
        else:
            if isinstance(imgsz, tuple):
                return (self._to_stride_multiple(int(imgsz[0])),
                        self._to_stride_multiple(int(imgsz[1])))
            return self._to_stride_multiple(int(imgsz))

    def _resolve_run_name(self, project_dir: Path, task: str, requested: str, auto_inc: bool) -> str:
        """
        Если имя пустое — сделать runN (ищем существующие).
        Если занято и auto_inc=True — добавим суффикс 2/3/...
        """
        name = requested.strip()
        task_dir = project_dir / self._task_to_subdir(task)
        task_dir.mkdir(parents=True, exist_ok=True)

        def exists(n: str) -> bool:
            return (task_dir / n).exists()

        if not name:
            # runN
            n = 1
            while exists(f"run{n}"):
                n += 1
            return f"run{n}"

        if exists(name) and auto_inc:
            k = 2
            while exists(f"{name}{k}"):
                k += 1
            return f"{name}{k}"
        return name

    @staticmethod
    def _task_to_subdir(task: str) -> str:
        """Как Ultralytics раскладывает по подпапкам."""
        return {"classifier": "classify", "detector": "detect", "segmentator": "segment"}.get(task, "detect")

    # -----------------------
    # Вспомогательные конвертеры
    # -----------------------

    def _state_to_dict(self) -> dict[str, Any]:
        imgsz = self._state.imgsz
        return {
            "model_name": self._state.model_name,
            "pretrained_model_path": self._state.model_path,
            "task_override": self._state.task_override,
            "profile": self._state.profile,
            "data": self._state.data,
            "project_dir": self._state.project_dir,
            "name": self._state.name,
            "auto_increment_name": self._state.auto_increment_name,
            "epochs": self._state.epochs,
            "patience": self._state.patience,
            "batch": self._state.batch,
            "imgsz": imgsz if isinstance(imgsz, int) else f"{imgsz[0]}x{imgsz[1]}",
            "rect": self._state.rect,
            "multi_scale": self._state.multi_scale,
            "seed": self._state.seed,
            "deterministic": self._state.deterministic,
            "workers": self._state.workers,
            "device": self._state.device,
        }

    def _state_from_dict(self, d: dict[str, Any]) -> AppState:
        imgsz = d.get("imgsz", 384)
        if isinstance(imgsz, str) and "x" in imgsz.lower():
            a, b = imgsz.lower().split("x", 1)
            imgsz_parsed: int | tuple[int, int] = (int(a), int(b))
        else:
            imgsz_parsed = int(imgsz)
        s = AppState(
            model_name=str(d.get("model_name", "")),
            model_path=str(d.get("pretrained_model_path", "")),
            task_override=str(d.get("task_override", "")),
            profile=str(d.get("profile", "fast")),  # валидация позже
            data=str(d.get("data", "")),
            project_dir=str(d.get("project_dir", self._paths.default_runs_dir.as_posix())),
            name=str(d.get("name", "")),
            auto_increment_name=bool(d.get("auto_increment_name", True)),
            epochs=int(d.get("epochs", 50)),
            patience=int(d.get("patience", 50)),
            batch=int(d.get("batch", 16)),
            imgsz=imgsz_parsed,
            rect=bool(d.get("rect", False)),
            multi_scale=bool(d.get("multi_scale", False)),
            seed=int(d.get("seed", 0)),
            deterministic=bool(d.get("deterministic", True)),
            workers=int(d.get("workers", 4)),
            device=str(d.get("device", "auto")),
        )
        return s
