from __future__ import annotations
from ..models_registry import ModelsRegistry

class TaskResolver:
    @staticmethod
    def resolve(models: ModelsRegistry, model_name: str, task_override: str) -> str:
        """
        Возвращает task: 'classifier' | 'detector' | 'segmentation'.
        Приоритет: task_override > по имени модели > 'detector' (по умолчанию).
        """
        if task_override:
            return task_override
        if model_name:
            t = models.infer_task_from_name(model_name)
            if t:
                return t
        return "detector"
