from __future__ import annotations

"""
    ModelsRegistry — хранит список поддерживаемых YOLO-моделей и умеет определять тип задачи.
    Поддержка:
    - YOLOv8, YOLOv9, YOLOv10, YOLOv11 (классификация, детекция, сегментация)
    - Автоопределение типа модели по имени файла
    - Получение списка доступных моделей для выпадающего списка в GUI
"""

from enum import Enum


class TaskType(str, Enum):
    """ Тип задачи YOLO. """
    CLASSIFY = "cls"       # Классификация
    DETECT = "det"         # Детекция
    SEGMENT = "seg"        # Сегментация
    UNKNOWN = "unknown"    # Не удалось определить


class ModelsRegistry:
    """ Реестр поддерживаемых YOLO-моделей. """

    # Формат: версия -> {тип: [списки моделей]}
    _MODELS = {
        "v8": {
            TaskType.CLASSIFY: ["yolov8n-cls.pt", "yolov8s-cls.pt", "yolov8m-cls.pt", "yolov8l-cls.pt", "yolov8x-cls.pt"],
            TaskType.DETECT:   ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
            TaskType.SEGMENT:  ["yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt", "yolov8x-seg.pt"],
        },
        "v9": {
            TaskType.CLASSIFY: ["yolov9c-cls.pt", "yolov9e-cls.pt"],
            TaskType.DETECT:   ["yolov9t.pt", "yolov9s.pt", "yolov9m.pt", "yolov9c.pt", "yolov9e.pt"],
            TaskType.SEGMENT:  ["yolov9t-seg.pt", "yolov9s-seg.pt", "yolov9m-seg.pt", "yolov9c-seg.pt", "yolov9e-seg.pt"],
        },
        "v10": {
            TaskType.CLASSIFY: ["yolo10n-cls.pt", "yolo10s-cls.pt", "yolo10m-cls.pt", "yolo10l-cls.pt", "yolo10x-cls.pt"],
            TaskType.DETECT:   ["yolo10n.pt", "yolo10s.pt", "yolo10m.pt", "yolo10l.pt", "yolo10x.pt"],
            TaskType.SEGMENT:  ["yolo10n-seg.pt", "yolo10s-seg.pt", "yolo10m-seg.pt", "yolo10l-seg.pt", "yolo10x-seg.pt"],
        },
        "v11": {
            TaskType.CLASSIFY: ["yolo11n-cls.pt", "yolo11s-cls.pt", "yolo11m-cls.pt", "yolo11l-cls.pt", "yolo11x-cls.pt"],
            TaskType.DETECT:   ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"],
            TaskType.SEGMENT:  ["yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt", "yolo11x-seg.pt"],
        }
    }

    @classmethod
    def get_all_models(cls) -> list[str]:
        """ Возвращает список всех моделей. """
        all_models = []
        for versions in cls._MODELS.values():
            for model_list in versions.values():
                all_models.extend(model_list)
        return all_models

    @classmethod
    def get_models_by_task(cls, task: TaskType) -> list[str]:
        """ Возвращает список моделей только для одного типа задачи. """
        result = []
        for versions in cls._MODELS.values():
            result.extend(versions.get(task, []))
        return result

    @classmethod
    def detect_task_from_name(cls, model_name: str) -> TaskType:
        """
            Определяет тип задачи по имени модели.
            Если имя не найдено в реестре — возвращает TaskType.UNKNOWN.
        """
        name = model_name.lower()
        for versions in cls._MODELS.values():
            for task, model_list in versions.items():
                if name in model_list:
                    return task
        return TaskType.UNKNOWN

    @classmethod
    def is_supported(cls, model_name: str) -> bool:
        """ Проверяет, есть ли модель в реестре. """
        return model_name.lower() in cls.get_all_models()
