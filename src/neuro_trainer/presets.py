from __future__ import annotations
"""
Presets — готовые профили гиперпараметров для задач YOLO (CLS/DET/SEG).
Совместимо с NeuroTrainerApp/Trainer:
  - PresetRegistry.resolve_name(task, profile) принимает:
      task: 'classifier' | 'detector' | 'segmentation' (а также синонимы: 'classification','segment','segmentation')
      profile: 'fast' | 'quality' | 'low_vram'
  - Preset.to_train_kwargs() -> dict, пригодный для передачи в model.train(...)
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(slots=True)
class Preset:
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

    # Аугментации/регуляризация (чаще для CLS)
    auto_augment: str | None = None
    erasing: float | None = None
    label_smoothing: float | None = None
    dropout: float | None = None

    def to_train_kwargs(self) -> Dict[str, Any]:
        """Только заданные в пресете (не None) параметры → dict для model.train(...)."""
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
    """CLS/DET/SEG × fast/quality/low_vram."""

    def __init__(self) -> None:
        self._presets: dict[str, Preset] = {}

        # -------------------
        # КЛАССИФИКАЦИЯ (CLS)
        # -------------------
        self._presets["cls_fast"] = Preset(
            epochs=20, patience=5, batch=32, imgsz=320,
            auto_augment="randaugment", erasing=0.15, label_smoothing=0.05,
            lr0=0.01, lrf=0.02, momentum=0.937, weight_decay=0.0005, warmup_epochs=2.0,
        )
        self._presets["cls_quality"] = Preset(
            epochs=80, patience=15, batch=32, imgsz=384,
            auto_augment="randaugment", erasing=0.30, label_smoothing=0.10, dropout=0.10,
            lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0,
        )
        self._presets["cls_low_vram"] = Preset(
            epochs=40, patience=10, batch=16, imgsz=256,
            auto_augment="randaugment", erasing=0.20, label_smoothing=0.05,
            lr0=0.008, lrf=0.02, momentum=0.937, weight_decay=0.0005, warmup_epochs=2.0,
        )

        # ---------------
        # ДЕТЕКЦИЯ (DET)
        # ---------------
        self._presets["det_fast"] = Preset(
            epochs=50, patience=10, batch=16, imgsz=640,
            rect=True, multi_scale=True,
            lr0=0.01, lrf=0.02, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0,
        )
        self._presets["det_quality"] = Preset(
            epochs=150, patience=30, batch=16, imgsz=800,
            rect=True, multi_scale=True,
            lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0,
        )
        self._presets["det_low_vram"] = Preset(
            epochs=80, patience=15, batch=8, imgsz=512,
            rect=True, multi_scale=True,
            lr0=0.008, lrf=0.02, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0,
        )

        # -------------------
        # СЕГМЕНТАЦИЯ (SEG)
        # -------------------
        self._presets["seg_fast"] = Preset(
            epochs=60, patience=12, batch=12, imgsz=640,
            rect=True, multi_scale=True,
            lr0=0.01, lrf=0.02, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0,
        )
        self._presets["seg_quality"] = Preset(
            epochs=180, patience=30, batch=12, imgsz=800,
            rect=True, multi_scale=True,
            lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0,
        )
        self._presets["seg_low_vram"] = Preset(
            epochs=100, patience=20, batch=6, imgsz=512,
            rect=True, multi_scale=True,
            lr0=0.008, lrf=0.02, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0,
        )

    # -------------------
    # Публичные методы
    # -------------------

    def names(self) -> list[str]:
        return sorted(self._presets.keys())

    def get(self, name: str) -> Preset:
        try:
            return self._presets[name]
        except KeyError as e:
            raise KeyError(f"Unknown preset '{name}'. Available: {', '.join(self.names())}") from e

    def resolve_name(self, task: str, profile: str) -> str:
        """
        task: 'classifier'|'detector'|'segmentation'
              (также принимаются 'classification','segment','segmentation')
        profile: 'fast'|'quality'|'low_vram'
        """
        t = task.strip().lower()
        p = profile.strip().lower()
        prefix_map = {
            "classifier": "cls", "classification": "cls",
            "detector": "det",
            "segmentation": "seg", "segment": "seg", "segmentation": "seg",
        }
        prefix = prefix_map.get(t)
        if not prefix:
            raise KeyError(f"Unknown task '{task}'. Expected classifier/detector/segmentation")
        key = f"{prefix}_{p}"
        if key not in self._presets:
            raise KeyError(f"Unknown profile '{profile}' for task '{task}'. Try fast/quality/low_vram")
        return key
