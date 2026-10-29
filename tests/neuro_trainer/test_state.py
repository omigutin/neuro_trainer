from neuro_trainer.state import AppState, ModelSource, TrainMode, state_from_dict, state_to_dict
from neuro_trainer.tasks import TaskKind


def test_roundtrip():
    s = AppState(model_source=ModelSource.CUSTOM, model_path="w.pt", train_mode=TrainMode.RESUME,
                 task=TaskKind.POSE, data="d.yaml", project_dir="runs", name="x")
    s.params["imgsz"] = (480, 640)
    d = state_to_dict(s)
    assert d["imgsz"] == "480x640" and d["task"] == "pose"
    back = state_from_dict(d, "runs")
    assert back == s


def test_legacy_format(caplog):
    old = {"model_name": "yolo11n-cls.pt", "pretrained_model_path": "", "task_override": "", "profile": "fast",
           "multi_scale": True, "label_smoothing": 0.05, "epochs": 0, "batch": 32, "imgsz": 320}
    s = state_from_dict(old, "runs")
    assert s.task is TaskKind.CLASSIFIER
    assert s.model_source is ModelSource.STANDARD
    assert s.params["multi_scale"] == 0.5
    assert s.params["batch"] == 32
    # ошибочное значение заменено значением по умолчанию
    assert s.params["epochs"] == 100
    assert s.project_dir == "runs"
    assert "label_smoothing" in caplog.text


def test_legacy_custom_path():
    s = state_from_dict({"pretrained_model_path": "C:/m/best.pt", "task_override": "segmentation"}, "runs")
    assert s.model_source is ModelSource.CUSTOM
    assert s.task is TaskKind.SEGMENTATION
