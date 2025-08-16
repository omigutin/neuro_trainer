from __future__ import annotations
"""
ModelsRegistry — список поддерживаемых YOLO-моделей и утилиты.
Совместимо с NeuroTrainerApp:
  - get_all_models()
  - is_known(name: str) -> bool
  - infer_task_from_name(name: str) -> 'classifier'|'detector'|'segmentation'
"""

from typing import List


class ModelsRegistry:
    # Версия -> {тип: [модели]}
    _MODELS = {
        "v8": {
            "classifier": ["yolov8n-cls.pt", "yolov8s-cls.pt", "yolov8m-cls.pt", "yolov8l-cls.pt", "yolov8x-cls.pt"],
            "detector":   ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
            "segmentation":["yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt", "yolov8x-seg.pt"],
        },
        "v9": {
            "classifier": ["yolov9c-cls.pt", "yolov9e-cls.pt"],
            "detector":   ["yolov9t.pt", "yolov9s.pt", "yolov9m.pt", "yolov9c.pt", "yolov9e.pt"],
            "segmentation":["yolov9t-seg.pt", "yolov9s-seg.pt", "yolov9m-seg.pt", "yolov9c-seg.pt", "yolov9e-seg.pt"],
        },
        "v10": {
            "classifier": ["yolo10n-cls.pt", "yolo10s-cls.pt", "yolo10m-cls.pt", "yolo10l-cls.pt", "yolo10x-cls.pt"],
            "detector":   ["yolo10n.pt", "yolo10s.pt", "yolo10m.pt", "yolo10l.pt", "yolo10x.pt"],
            "segmentation":["yolo10n-seg.pt", "yolo10s-seg.pt", "yolo10m-seg.pt", "yolo10l-seg.pt", "yolo10x-seg.pt"],
        },
        "v11": {
            "classifier": ["yolo11n-cls.pt", "yolo11s-cls.pt", "yolo11m-cls.pt", "yolo11l-cls.pt", "yolo11x-cls.pt"],
            "detector":   ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"],
            "segmentation":["yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt", "yolo11x-seg.pt"],
        },
    }

    @classmethod
    def get_all_models(cls) -> List[str]:
        out: List[str] = []
        for versions in cls._MODELS.values():
            for lst in versions.values():
                out.extend(lst)
        return out

    @classmethod
    def is_known(cls, name: str) -> bool:
        return name.lower() in (m.lower() for m in cls.get_all_models())

    @classmethod
    def infer_task_from_name(cls, name: str) -> str:
        n = name.lower()
        if "cls" in n:
            return "classifier"
        if "seg" in n:
            return "segmentation"
        # иначе считаем детектором (yolo*.pt)
        return "detector"

    @classmethod
    def get_models_by_task(cls, task: str) -> List[str]:
        t = task.lower()
        out: List[str] = []
        for versions in cls._MODELS.values():
            out.extend(versions.get(t, []))
        return out
