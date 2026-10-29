from neuro_trainer.job import JobKind, JobSpec, JobState, JobStatus
from neuro_trainer.tasks import TaskKind


def test_spec_roundtrip(tmp_path):
    spec = JobSpec(kind=JobKind.VALIDATE, task=TaskKind.POSE, data="d", project_dir=str(tmp_path), run_name="r",
                   model_name="yolo26n-pose.pt", train_args={"imgsz": 640})
    spec.save(tmp_path / "s.json")
    assert JobSpec.load(tmp_path / "s.json") == spec
    assert spec.run_dir == tmp_path.resolve() / "pose" / "r"


def test_status_roundtrip(tmp_path):
    assert JobStatus.load(tmp_path) is None
    JobStatus(state=JobState.STOPPED, kind=JobKind.TRAIN, error="e").save(tmp_path)
    st = JobStatus.load(tmp_path)
    assert st.state is JobState.STOPPED and not st.state.active and st.error == "e"
