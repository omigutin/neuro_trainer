from neuro_trainer.services.imgsz import normalize_imgsz
from neuro_trainer.tasks import TaskKind


def test_normalize():
    assert normalize_imgsz(500, TaskKind.DETECTOR) == 512
    assert normalize_imgsz((470, 650), TaskKind.CLASSIFIER) == 640
    assert normalize_imgsz((470, 650), TaskKind.POSE) == (480, 640)
    assert normalize_imgsz(5, TaskKind.OBB) == 32
