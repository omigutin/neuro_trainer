from __future__ import annotations

"""
    Presets — готовые профили гиперпараметров для разных задач YOLO (CLS/DET/SEG).
    
    Идея:
    - Упростить выбор настроек: пользователь выбирает тип задачи (classifier/detector/segmentator)
      и профиль (fast/quality/low_vram), а мы подставляем вменяемые значения обучения.
    - Все значения — «разумные по умолчанию». Это НЕ серебряная пуля. Их можно править в GUI.
    
    Профили:
      fast
        • Цель: «быстро проверить гипотезу». Меньше эпох, умеренные аугментации.
        • Когда: первичная проверка датасета/идей, экономия времени.
    
      quality
        • Цель: «выжать качество». Дольше обучаем, чуть сильнее аугментации/регуляризации.
        • Когда: уже уверены в данных, нужно максимум точности.
    
      low_vram
        • Цель: «не падать по видеопамяти». Уменьшаем batch и imgsz.
        • Когда: слабый GPU (4–6 ГБ), частые OOM (Out Of Memory).
    
    Примечания:
    - imgsz: реальное «округление к кратности 32» выполнит фасад (NeuroTrainerApp/Trainer).
    - Для классификации imgsz ожидается как int (квадрат). Для детекции/сегментации — int или (H, W).
    - Параметры lr0/lrf/momentum/weight_decay/warmup_epochs — стандартные для Ultralytics;
      они подходят как хороший старт.
    
    Согласовано с NeuroTrainerApp:
    - PresetRegistry.resolve_name(task, profile) -> "cls_fast"/"det_quality"/"seg_low_vram", и т.п.
    - PresetRegistry.get(name) -> Preset
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(slots=True)
class Preset:
    """
        Набор гиперпараметров для передачи в Trainer/Ultralytics.
        Все поля опциональны — задаём только то, что хотим переопределить.
        Остальное придёт из пользовательских настроек (GUI) и дефолтов.
    """
    # Общие
    epochs: int | None = None
    patience: int | None = None
    batch: int | None = None
    imgsz: int | tuple[int, int] | None = None
    verbose: bool | None = True

    # Детекция/сегментация
    rect: bool | None = None
    multi_scale: bool | None = None

    # Оптимизация
    lr0: float | None = None
    lrf: float | None = None
    momentum: float | None = None
    weight_decay: float | None = None
    warmup_epochs: float | None = None

    # Аугментации/регуляризация (прежде всего полезно для CLS)
    auto_augment: str | None = None
    erasing: float | None = None
    label_smoothing: float | None = None
    dropout: float | None = None

    def as_train_kwargs(self) -> Dict[str, Any]:
        """
            Возвращает только те параметры, которые явно заданы в пресете (не None).
            Это удобно передавать напрямую в model.train(...).
        """
        fields = (
            "epochs", "patience", "batch", "imgsz", "verbose",
            "rect", "multi_scale",
            "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
            "auto_augment", "erasing", "label_smoothing", "dropout",
        )
        out: Dict[str, Any] = {}
        for k in fields:
            v = getattr(self, k)
            if v is not None:
                out[k] = v
        return out


class PresetRegistry:
    """
        Реестр всех пресетов: CLS/DET/SEG × fast/quality/low_vram.
        Использование:
            reg = PresetRegistry()
            name = reg.resolve_name("classifier", "fast")  # -> "cls_fast"
            preset = reg.get(name)
            kwargs = preset.as_train_kwargs()
    """

    def __init__(self) -> None:
        self._presets: dict[str, Preset] = {}

        # -------------------
        # КЛАССИФИКАЦИЯ (CLS)
        # -------------------
        # fast — быстро проверить гипотезу; умеренные параметры
        self._presets["cls_fast"] = Preset(
            epochs=20, patience=5, batch=32, imgsz=320,
            auto_augment="randaugment", erasing=0.15, label_smoothing=0.05,
            lr0=0.01, lrf=0.02, momentum=0.937, weight_decay=0.0005, warmup_epochs=2.0,
        )

        # quality — больше эпох, чуть сильнее регуляризация
        self._presets["cls_quality"] = Preset(
            epochs=80, patience=15, batch=32, imgsz=384,
            auto_augment="randaugment", erasing=0.30, label_smoothing=0.10, dropout=0.10,
            lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0,
        )

        # low_vram — уменьшенные imgsz и batch, чтобы не падать по памяти
        self._presets["cls_low_vram"] = Preset(
            epochs=40, patience=10, batch=16, imgsz=256,
            auto_augment="randaugment", erasing=0.20, label_smoothing=0.05,
            lr0=0.008, lrf=0.02, momentum=0.937, weight_decay=0.0005, warmup_epochs=2.0,
        )

        # ---------------
        # ДЕТЕКЦИЯ (DET)
        # ---------------
        # fast — стартовый баланс; rect/multi_scale для робастности
        self._presets["det_fast"] = Preset(
            epochs=50, patience=10, batch=16, imgsz=640,
            rect=True, multi_scale=True,
            lr0=0.01, lrf=0.02, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0,
        )

        # quality — больше эпох и более детальный imgsz
        self._presets["det_quality"] = Preset(
            epochs=150, patience=30, batch=16, imgsz=800,
            rect=True, multi_scale=True,
            lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0,
        )

        # low_vram — снизили imgsz и batch для экономии VRAM
        self._presets["det_low_vram"] = Preset(
            epochs=80, patience=15, batch=8, imgsz=512,
            rect=True, multi_scale=True,
            lr0=0.008, lrf=0.02, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0,
        )

        # -------------------
        # СЕГМЕНТАЦИЯ (SEG)
        # -------------------
        # fast — стартовый вариант; параметры близки к детекции
        self._presets["seg_fast"] = Preset(
            epochs=60, patience=12, batch=12, imgsz=640,
            rect=True, multi_scale=True,
            lr0=0.01, lrf=0.02, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0,
        )

        # quality — больше эпох и выше imgsz
        self._presets["seg_quality"] = Preset(
            epochs=180, patience=30, batch=12, imgsz=800,
            rect=True, multi_scale=True,
            lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0,
        )

        # low_vram — облегчённый вариант под слабые GPU
        self._presets["seg_low_vram"] = Preset(
            epochs=100, patience=20, batch=6, imgsz=512,
            rect=True, multi_scale=True,
            lr0=0.008, lrf=0.02, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0,
        )

    # -------------------
    # Публичные методы
    # -------------------

    def names(self) -> list[str]:
        """ Возвращает список всех доступных имён пресетов. """
        return sorted(self._presets.keys())

    def get(self, name: str) -> Preset:
        """ Возвращает пресет по имени or бросает KeyError, если не найден. """
        try:
            return self._presets[name]
        except KeyError as e:
            raise KeyError(f"Unknown preset '{name}'. Available: {', '.join(self.names())}") from e

    def resolve_name(self, task: str, profile: str) -> str:
        """
            Преобразует («тип задачи» × «профиль») → имя пресета.
            task: "classification" | "detector" | "segmentation"
            profile: "fast" | "quality" | "low_vram"
        """
        t = task.strip().lower()
        p = profile.strip().lower()
        prefix = {"classification": "cls", "detector": "det", "segmentation": "seg"}.get(t)
        if not prefix:
            raise KeyError(f"Unknown task '{task}'. Expected classification/detector/segmentation")
        key = f"{prefix}_{p}"
        if key not in self._presets:
            raise KeyError(f"Unknown profile '{profile}' for task '{task}'. "
                           f"Try one of: fast/quality/low_vram")
        return key
