from pathlib import Path

import pytest

from neuro_trainer.config_store import ConfigStore
from neuro_trainer.job import JobKind
from neuro_trainer.neuro_trainer import NeuroTrainerApp
from neuro_trainer.presets import PresetRegistry
from neuro_trainer.state import Profile
from neuro_trainer.tasks import TaskKind


@pytest.fixture
def app(paths):
    return NeuroTrainerApp(paths, ConfigStore(paths), PresetRegistry(paths.presets_dir))


def test_update_standard_model_sets_task_and_reports_errors(app):
    errors = app.update({"model_name": "yolo26s-pose.pt", "epochs": "20", "lr0": "abc"})
    assert set(errors) == {"lr0"}
    st = app.state_dict()
    assert st["task"] == "pose" and st["epochs"] == 20


def test_update_unknown_standard_model(app):
    assert "model_name" in app.update({"model_name": "yolo99n.pt"})


def test_build_job_standard(app, tmp_path):
    data = tmp_path / "data.yaml"
    data.write_text("names: [a]")
    app.update({"model_name": "yolo11n-seg.pt", "data": str(data), "imgsz": "500", "copy_paste": 0.3,
                "dropout": 0.2, "device": "auto"})
    spec = app.build_job(JobKind.TRAIN)
    assert spec.task is TaskKind.SEGMENTATION and spec.model_name == "yolo11n-seg.pt" and not spec.model_path
    assert spec.run_name == "run1" and spec.data == str(data.resolve())
    assert spec.train_args["imgsz"] == 512 and spec.train_args["copy_paste"] == 0.3
    assert "dropout" not in spec.train_args and "device" not in spec.train_args
    assert spec.run_dir == Path(spec.project_dir) / "segment" / "run1"


def test_build_job_requires_data(app):
    app.update({"data": ""})
    with pytest.raises(ValueError, match="датасет"):
        app.build_job(JobKind.TRAIN)


def test_custom_finetune_and_resume(app, tmp_path):
    run = tmp_path / "old" / "detect" / "run3" / "weights"
    run.mkdir(parents=True)
    best, last = run / "best.pt", run / "last.pt"
    best.write_text("w")
    last.write_text("w")
    app.update({"model_source": "custom", "model_path": str(best), "task": "detector", "data": "coco8.yaml"})
    spec = app.build_job(JobKind.TRAIN)
    assert spec.model_path == str(best.resolve()) and not spec.resume and spec.data == "coco8.yaml"

    app.update({"train_mode": "resume"})
    with pytest.raises(ValueError, match="last.pt"):
        app.build_job(JobKind.TRAIN)
    app.update({"model_path": str(last), "data": ""})
    spec = app.build_job(JobKind.TRAIN)
    assert spec.resume and spec.run_dir == run.parent.resolve()


def test_custom_missing_file(app):
    app.update({"model_source": "custom", "model_path": "nope.pt", "data": "d.yaml"})
    with pytest.raises(ValueError, match="не найден"):
        app.build_job(JobKind.TRAIN)


def test_inspect_custom_model(app, tmp_path, fake_ultralytics):
    w = tmp_path / "parts-seg.pt"
    w.write_text("w")
    info = app.inspect_custom_model(str(w))
    assert info.task is TaskKind.SEGMENTATION and info.class_names == {0: "a", 1: "b"} and info.imgsz == 320
    st = app.state_dict()
    assert st["model_source"] == "custom" and st["task"] == "segmentation"


def test_profiles_and_user_presets(app):
    values = app.profile_values(TaskKind.CLASSIFIER, Profile.QUALITY)
    assert values["imgsz"] == 384
    app.update({"epochs": 7})
    app.save_user_preset("my")
    app.update({"epochs": 9})
    app.load_user_preset("my")
    assert app.state_dict()["epochs"] == 7
    assert app.list_user_presets() == ["my"]
    with pytest.raises(ValueError):
        app.save_user_preset("../x")


def test_last_session(app, paths):
    app.update({"epochs": 11})
    app.save_last_session()
    other = NeuroTrainerApp(paths, ConfigStore(paths), PresetRegistry(paths.presets_dir))
    other.load_last_session()
    assert other.state_dict()["epochs"] == 11
