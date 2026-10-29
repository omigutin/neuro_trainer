import time

import pytest

from neuro_trainer.job import JobKind, JobSpec, JobState
from neuro_trainer.job_manager import JobBusyError, JobManager
from neuro_trainer.tasks import TaskKind


def _spec(tmp_path, **kw):
    weights = tmp_path / "w" / "yolo26n.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_text("w")
    args = dict(kind=JobKind.TRAIN, task=TaskKind.DETECTOR, data="d.yaml", project_dir=str(tmp_path / "runs"),
                run_name="run1", model_name="yolo26n.pt", weights_dir=str(weights.parent),
                train_args={"epochs": 3})
    args.update(kw)
    return JobSpec(**args)


def _wait(jm, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        view = jm.status()
        if not view.state.active:
            return view
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def test_train_job_finishes(tmp_path, fake_ultralytics):
    jm = JobManager()
    view = jm.start(_spec(tmp_path))
    assert view.state.active
    view = _wait(jm)
    assert view.state is JobState.FINISHED, jm.log_tail()
    assert [r["epoch"] for r in jm.progress()] == [1, 2, 3]
    assert [r["epoch"] for r in jm.progress(since=2)] == [3]
    assert "fake training done" in jm.log_tail()


def test_failed_job_reports_error(tmp_path, fake_ultralytics):
    jm = JobManager()
    jm.start(_spec(tmp_path, model_name="", model_path=str(tmp_path / "missing.pt")))
    view = _wait(jm)
    assert view.state is JobState.FAILED
    assert "missing.pt" in view.error


def test_stop_and_busy(tmp_path, fake_ultralytics, monkeypatch):
    monkeypatch.setenv("FAKE_UL_EPOCH_SLEEP", "0.5")
    jm = JobManager()
    jm.start(_spec(tmp_path, train_args={"epochs": 50}))
    with pytest.raises(JobBusyError):
        jm.start(_spec(tmp_path, run_name="run2"))
    while not jm.progress():
        time.sleep(0.1)
    jm.request_stop()
    view = _wait(jm)
    assert view.state is JobState.STOPPED
    assert len(jm.progress()) < 50


def test_kill(tmp_path, fake_ultralytics, monkeypatch):
    monkeypatch.setenv("FAKE_UL_EPOCH_SLEEP", "1")
    jm = JobManager()
    jm.start(_spec(tmp_path, train_args={"epochs": 50}))
    while not jm.progress():
        time.sleep(0.1)
    jm.kill()
    assert jm.status().state is JobState.KILLED
    with pytest.raises(RuntimeError):
        jm.request_stop()


def test_validate_job(tmp_path, fake_ultralytics):
    jm = JobManager()
    jm.start(_spec(tmp_path, kind=JobKind.VALIDATE))
    assert _wait(jm).state is JobState.FINISHED
