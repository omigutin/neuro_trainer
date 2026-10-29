import pytest

from neuro_trainer.models_registry import ModelsRegistry
from neuro_trainer.tasks import TaskKind


def test_yolo26_all_tasks_and_sizes():
    fam = ModelsRegistry.get_family("yolo26")
    assert fam.models(TaskKind.DETECTOR) == [f"yolo26{s}.pt" for s in "nsmlx"]
    assert fam.models(TaskKind.CLASSIFIER)[0] == "yolo26n-cls.pt"
    assert fam.models(TaskKind.SEGMENTATION)[-1] == "yolo26x-seg.pt"
    assert fam.models(TaskKind.POSE)[2] == "yolo26m-pose.pt"
    assert fam.models(TaskKind.OBB)[3] == "yolo26l-obb.pt"


@pytest.mark.parametrize("name,task", [
    ("yolo26n.pt", TaskKind.DETECTOR), ("yolo11s-cls.pt", TaskKind.CLASSIFIER),
    ("yolov9e-seg.pt", TaskKind.SEGMENTATION), ("yolov8x-pose.pt", TaskKind.POSE),
    ("yolo26n-obb.pt", TaskKind.OBB), ("yolov10b.pt", TaskKind.DETECTOR),
])
def test_infer_task(name, task):
    assert ModelsRegistry.is_known(name)
    assert ModelsRegistry.infer_task_from_name(name) is task


@pytest.mark.parametrize("name", ["yolo10n-cls.pt", "yolov9c-cls.pt", "my_model.pt"])
def test_unknown_names(name):
    assert not ModelsRegistry.is_known(name)
    with pytest.raises(ValueError):
        ModelsRegistry.infer_task_from_name(name)


def test_find_family():
    assert ModelsRegistry.find_family("yolo11n-seg.pt").key == "yolo11"
