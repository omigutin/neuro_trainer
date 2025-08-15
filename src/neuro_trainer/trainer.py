# src/neuro_trainer/trainer.py
from __future__ import annotations

"""
Trainer — тонкая обёртка над Ultralytics YOLO для обучения/валидации.

Что добавлено:
- Колбэки:
  * Ранний стоп по top1 для классификации.
  * Ранний стоп по mAP50-95 для детекции/сегментации.
  * JSONL-лог по эпохам (train_metrics.jsonl).
- Автоподхват task из модели, нормализация imgsz (кратно 32; CLS → квадрат).
- Сохранение итоговых метрик (final_metrics.json / final_metrics_extra.csv).

Параметры для колбэков:
- jsonl_log: включить/выключить JSONL-лог (по эпохам).
- early_stop_cls_threshold, early_stop_cls_patience: порог и «терпение» для CLS.
- early_stop_map_threshold, early_stop_map_patience: порог и «терпение» для DET/SEG.

Замечание:
Ultralytics устанавливает `model.trainer.save_dir` уже в начале fit-цикла.
Мы создаём логгер «лениво»: при первом вызове колбэка берём актуальный save_dir.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ultralytics import YOLO

from .presets import Preset
from .metrics import MetricsStore
from .callbacks import EarlyStopOnTop1, EarlyStopOnMap, JsonlLogger, StopOnEvent


@dataclass(slots=True)
class Trainer:
    # Источник весов: имя стандартной модели (yolo11n.pt) ИЛИ абсолютный путь к локальному .pt
    weights: str
    # Подсказка по типу задачи (может быть пустой: "", тогда определяем из веса)
    task_hint: str
    # Данные
    data: str
    # Куда писать результаты
    project_dir: str
    run_name: str

    # Гиперпараметры/флаги
    epochs: int
    patience: int
    batch: int
    imgsz: int | tuple[int, int]
    rect: bool
    multi_scale: bool
    seed: int
    deterministic: bool
    workers: int
    device: str  # "auto"|"cuda"|"mps"|"cpu"

    # Пресет (добавляет/переопределяет часть полей в train())
    preset: Optional[Preset] = None

    # Колбэки/логирование
    jsonl_log: bool = True
    early_stop_cls_threshold: float = 0.95
    early_stop_cls_patience: int = 3
    early_stop_map_threshold: float = 0.55
    early_stop_map_patience: int = 5

    # Выходные поля (заполняются после запуска)
    save_dir: Optional[str] = None
    task: Optional[str] = None
    jsonl_log_path: Optional[Path] = None  # итоговый путь к train_metrics.jsonl

    # ------------------------
    # Публичные методы
    # ------------------------

    def train(self) -> None:
        """Полный цикл обучения + сохранение итоговых метрик."""
        model = YOLO(self.weights)

        # Определяем реальный тип задачи у модели
        real_task = getattr(model, "task", None) or ""
        self.task = self._normalize_task_name(self.task_hint or real_task or "detect")

        # Подготовка аргументов обучения
        train_kwargs = self._build_train_kwargs_for_model(model, self.task)

        # Прицепим колбэки
        self._attach_callbacks(model, self.task)

        # Запуск обучения
        model.train(**train_kwargs)

        # Где лежит результат рана
        self.save_dir = str(Path(model.trainer.save_dir))

        # Итоговые метрики — единообразно после val()
        metrics = model.val()
        MetricsStore(Path(self.save_dir)).save(metrics, self.task)

    def validate_only(self) -> None:
        """Быстрая валидация датасета/модели без обучения (полезно перед долгим train)."""
        model = YOLO(self.weights)

        real_task = getattr(model, "task", None) or ""
        self.task = self._normalize_task_name(self.task_hint or real_task or "detect")

        # Валидация с нашими параметрами
        _, imgsz_norm = self._normalize_imgsz(self.imgsz, self.task)
        val_results = model.val(data=self.data, imgsz=imgsz_norm, batch=self.batch)

        self.save_dir = str(Path(model.trainer.save_dir))
        MetricsStore(Path(self.save_dir)).save(val_results, self.task)

    # ------------------------
    # Внутренняя кухня
    # ------------------------

    def _build_train_kwargs_for_model(self, model: YOLO, task: str) -> Dict[str, Any]:
        """Собирает kwargs для model.train, склеивая пользовательские параметры и пресет."""
        # База
        kwargs: Dict[str, Any] = dict(
            data=self.data,
            project=self.project_dir,
            name=self.run_name,
            epochs=self.epochs,
            patience=self.patience,
            batch=self.batch,
            verbose=True,                # подробные логи
            seed=self.seed,
            deterministic=self.deterministic,
            workers=self.workers,
        )

        # imgsz: нормализуем (кратно 32; CLS -> квадратный int)
        rect_flag, imgsz_norm = self._normalize_imgsz(self.imgsz, task)
        kwargs["imgsz"] = imgsz_norm

        # флаги детекции/сегментации
        if task in {"detector", "segmentator"}:
            kwargs["rect"] = self.rect or rect_flag
            kwargs["multi_scale"] = self.multi_scale

        # Пресет (перекрывает базу только явно заданными полями)
        if self.preset is not None:
            for k, v in self.preset.as_train_kwargs().items():
                if k == "imgsz":
                    _, v = self._normalize_imgsz(v, task)
                kwargs[k] = v

        # Устройство
        device = self._select_device(self.device)
        model.to(device)

        return kwargs

    def _attach_callbacks(self, model: YOLO, task: str) -> None:
        """
        Подключает колбэки Ultralytics:
        - ранний стоп по top1/mAP (если пороги заданы разумно);
        - JSONL-лог (если включён).
        Логгер создаётся «лениво» — на первом вызове берём актуальный save_dir.
        """
        # мягкая отмена из GUI
        model.add_callback("on_fit_epoch_end", StopOnEvent())

        # ранний стоп
        if task == "classifier" and self.early_stop_cls_threshold > 0 and self.early_stop_cls_patience > 0:
            model.add_callback("on_fit_epoch_end",
                               EarlyStopOnTop1(self.early_stop_cls_threshold, self.early_stop_cls_patience))
        if task in {"detector",
                    "segmentator"} and self.early_stop_map_threshold > 0 and self.early_stop_map_patience > 0:
            model.add_callback("on_fit_epoch_end",
                               EarlyStopOnMap(self.early_stop_map_threshold, self.early_stop_map_patience))

        # JSONL-лог — ленивый инициализатор (как было)
        if self.jsonl_log:
            app = self

            class _LazyJsonl:
                def __init__(self) -> None:
                    self._logger: Optional[JsonlLogger] = None

                def __call__(self, trainer_obj: Any) -> None:
                    if self._logger is None:
                        save_dir = Path(getattr(trainer_obj, "save_dir",
                                                getattr(getattr(trainer_obj, "model", None), "save_dir", ".")))
                        save_dir = Path(save_dir);
                        save_dir.mkdir(parents=True, exist_ok=True)
                        log_path = save_dir / "train_metrics.jsonl"
                        app.jsonl_log_path = log_path
                        self._logger = JsonlLogger(log_path)
                    self._logger(trainer_obj)

            model.add_callback("on_fit_epoch_end", _LazyJsonl())

    @staticmethod
    def _normalize_task_name(task: str) -> str:
        t = (task or "").strip().lower()
        if t in ("cls", "classify", "classifier"):
            return "classifier"
        if t in ("det", "detect", "detector"):
            return "detector"
        if t in ("seg", "segment", "segmentor", "segmentator"):
            return "segmentator"
        # дефолт — считаем детекцией
        return "detector"

    def _normalize_imgsz(self, imgsz: int | tuple[int, int], task: str) -> tuple[bool, int | tuple[int, int]]:
        """
        Возвращает (rect_flag, imgsz_norm).

        - Классификация: только int (квадрат), округляем до кратного 32.
        - Детекция/Сегментация: int или (H, W), каждое округляем до кратного 32.
        - rect_flag: для классификации False; для дет/сег — не навязываем (False).
        """
        def to_stride_multiple(x: int, stride: int = 32) -> int:
            return max(32, int(round(x / stride) * stride))

        if task == "classifier":
            if isinstance(imgsz, tuple):
                imgsz = max(int(imgsz[0]), int(imgsz[1]))
            return False, to_stride_multiple(int(imgsz))

        # detector/segmentator
        if isinstance(imgsz, tuple):
            h = to_stride_multiple(int(imgsz[0]))
            w = to_stride_multiple(int(imgsz[1]))
            return False, (h, w)
        return False, to_stride_multiple(int(imgsz))

    @staticmethod
    def _select_device(device: str) -> str:
        """
        Упрощённый выбор устройства.
        Ultralytics сам корректно обработает 'cuda', 'mps', 'cpu'. 'auto' — оставляем на совесть окружения.
        """
        dev = (device or "auto").lower().strip()
        if dev in ("auto", "cuda", "mps", "cpu"):
            return dev
        return "auto"
