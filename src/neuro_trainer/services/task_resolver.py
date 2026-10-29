from __future__ import annotations

from ..models_registry import ModelsRegistry
from ..tasks import TaskKind


class TaskResolver:
    @staticmethod
    def resolve(model_name: str, fallback: TaskKind) -> TaskKind:
        """
        Задача для стандартных весов — по имени; для прочих имён — fallback
        (задача своих весов определяется из файла через model_inspector).
        """
        if model_name and ModelsRegistry.is_known(model_name):
            return ModelsRegistry.infer_task_from_name(model_name)
        return fallback
