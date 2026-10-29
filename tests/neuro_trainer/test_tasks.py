import pytest

from neuro_trainer.tasks import TASKS, TaskKind, parse_task, task_info


def test_parse_internal_and_ultralytics_names():
    assert parse_task("classifier") is TaskKind.CLASSIFIER
    assert parse_task("classify") is TaskKind.CLASSIFIER
    assert parse_task("segment") is TaskKind.SEGMENTATION
    assert parse_task(" OBB ") is TaskKind.OBB
    with pytest.raises(ValueError):
        parse_task("depth")


def test_every_task_described():
    assert set(TASKS) == set(TaskKind)
    assert task_info("pose").ultralytics_task == "pose"
