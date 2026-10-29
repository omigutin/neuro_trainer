from __future__ import annotations
"""
ModelsRegistry — стандартные веса Ultralytics, которые можно скачать по имени.

Список соответствует GITHUB_ASSETS_NAMES Ultralytics 8.4 (ultralytics/utils/downloads.py):
задачи и размеры, для которых опубликованы готовые .pt.
"""

from dataclasses import dataclass

from .tasks import TaskKind, TASKS


@dataclass(frozen=True, slots=True)
class ModelFamily:
    """ Линейка моделей: префикс имени весов и доступные размеры по задачам. """
    # Идентификатор линейки (значение в форме)
    key: str
    # Подпись в интерфейсе
    label: str
    # Префикс имени файла: <prefix><size><suffix>.pt
    prefix: str
    # Доступные размеры по задачам (порядок — от меньшей модели к большей)
    sizes: dict[TaskKind, str]

    def models(self, task: TaskKind) -> list[str]:
        """ Имена весов линейки для задачи. """
        suffix = TASKS[task].weights_suffix
        return [f"{self.prefix}{size}{suffix}.pt" for size in self.sizes.get(task, "")]


_ALL_TASKS_NSMLX = {task: "nsmlx" for task in TaskKind}

FAMILIES: tuple[ModelFamily, ...] = (
    ModelFamily("yolo26", "YOLO26", "yolo26", _ALL_TASKS_NSMLX),
    ModelFamily("yolo11", "YOLO11", "yolo11", _ALL_TASKS_NSMLX),
    ModelFamily("yolo12", "YOLO12", "yolo12", {TaskKind.DETECTOR: "nsmlx"}),
    ModelFamily("yolov10", "YOLOv10", "yolov10", {TaskKind.DETECTOR: "nsmblx"}),
    ModelFamily("yolov9", "YOLOv9", "yolov9", {TaskKind.DETECTOR: "tsmce", TaskKind.SEGMENTATION: "ce"}),
    ModelFamily("yolov8", "YOLOv8", "yolov8", _ALL_TASKS_NSMLX),
)


class ModelsRegistry:
    """ Поиск стандартных весов по линейке и задаче. """

    @staticmethod
    def families() -> tuple[ModelFamily, ...]:
        """ Все линейки (новые сверху). """
        return FAMILIES

    @staticmethod
    def get_family(key: str) -> ModelFamily:
        """ Линейка по идентификатору. """
        for family in FAMILIES:
            if family.key == key:
                return family
        raise KeyError(f"Unknown model family '{key}'")

    @staticmethod
    def get_all_models() -> list[str]:
        """ Все стандартные веса всех линеек и задач. """
        return [m for family in FAMILIES for task in TaskKind for m in family.models(task)]

    @classmethod
    def is_known(cls, name: str) -> bool:
        """ Является ли имя стандартными весами Ultralytics. """
        return name.lower() in {m.lower() for m in cls.get_all_models()}

    @classmethod
    def infer_task_from_name(cls, name: str) -> TaskKind:
        """ Задача по имени стандартных весов. """
        n = name.lower()
        for family in FAMILIES:
            for task in TaskKind:
                if n in (m.lower() for m in family.models(task)):
                    return task
        raise ValueError(f"'{name}' is not a standard Ultralytics weights name")

    @staticmethod
    def get_models_by_task(task: TaskKind) -> list[str]:
        """ Стандартные веса всех линеек для задачи. """
        return [m for family in FAMILIES for m in family.models(task)]

    @staticmethod
    def find_family(name: str) -> ModelFamily:
        """ Линейка, к которой относятся стандартные веса. """
        n = name.lower()
        for family in FAMILIES:
            if any(n == m.lower() for task in TaskKind for m in family.models(task)):
                return family
        raise ValueError(f"'{name}' is not a standard Ultralytics weights name")
