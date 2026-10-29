from __future__ import annotations
"""
Типы задач обучения и их соответствие понятиям Ultralytics.

Внутреннее имя задачи (TaskKind.value) хранится в состоянии и YAML-пресетах;
имя задачи Ultralytics используется при обучении и в названии подпапки результатов.
"""

from dataclasses import dataclass
from enum import Enum


class TaskKind(str, Enum):
    """ Задача обучения. Значение — внутреннее имя (совместимо со старыми last_session.yaml). """
    CLASSIFIER = "classifier"
    DETECTOR = "detector"
    SEGMENTATION = "segmentation"
    POSE = "pose"
    OBB = "obb"


@dataclass(frozen=True, slots=True)
class TaskInfo:
    """ Описание задачи: имена в Ultralytics, суффикс весов, основная метрика для графика. """
    # Внутренний тип задачи
    kind: TaskKind
    # Подпись в интерфейсе
    label: str
    # Имя задачи в Ultralytics (task=...), оно же подпапка результатов: runs/<ultralytics_task>/<run>
    ultralytics_task: str
    # Суффикс имени стандартных весов: yolo11n<suffix>.pt
    weights_suffix: str
    # Ключ основной метрики в trainer.metrics (ось графика прогресса)
    main_metric: str
    # Префикс YAML-пресетов: configs/presets/<preset_prefix>_<profile>.yaml
    preset_prefix: str


TASKS: dict[TaskKind, TaskInfo] = {
    TaskKind.CLASSIFIER: TaskInfo(TaskKind.CLASSIFIER, "Классификация", "classify", "-cls",
                                  "metrics/accuracy_top1", "cls"),
    TaskKind.DETECTOR: TaskInfo(TaskKind.DETECTOR, "Детекция", "detect", "",
                                "metrics/mAP50-95(B)", "det"),
    TaskKind.SEGMENTATION: TaskInfo(TaskKind.SEGMENTATION, "Сегментация", "segment", "-seg",
                                    "metrics/mAP50-95(M)", "seg"),
    TaskKind.POSE: TaskInfo(TaskKind.POSE, "Ключевые точки (pose)", "pose", "-pose",
                            "metrics/mAP50-95(P)", "pose"),
    TaskKind.OBB: TaskInfo(TaskKind.OBB, "Повёрнутые рамки (OBB)", "obb", "-obb",
                           "metrics/mAP50-95(B)", "obb"),
}

_BY_ULTRALYTICS: dict[str, TaskKind] = {info.ultralytics_task: kind for kind, info in TASKS.items()}


def parse_task(value: str | TaskKind) -> TaskKind:
    """ Внутреннее имя задачи или имя Ultralytics → TaskKind. """
    if isinstance(value, TaskKind):
        return value
    v = (value or "").strip().lower()
    if v in _BY_ULTRALYTICS:
        return _BY_ULTRALYTICS[v]
    try:
        return TaskKind(v)
    except ValueError:
        raise ValueError(f"Unknown task '{value}'. Expected one of: {', '.join(t.value for t in TaskKind)}") from None


def task_info(task: str | TaskKind) -> TaskInfo:
    """ Описание задачи по внутреннему имени или имени Ultralytics. """
    return TASKS[parse_task(task)]
