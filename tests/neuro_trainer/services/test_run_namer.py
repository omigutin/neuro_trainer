from neuro_trainer.services.run_namer import resolve_run_name
from neuro_trainer.tasks import TaskKind


def test_run_names(tmp_path):
    assert resolve_run_name(tmp_path, TaskKind.OBB, "") == "run1"
    (tmp_path / "obb" / "run1").mkdir()
    assert resolve_run_name(tmp_path, TaskKind.OBB, "") == "run2"
    (tmp_path / "obb" / "exp").mkdir()
    assert resolve_run_name(tmp_path, TaskKind.OBB, "exp") == "exp2"
    assert resolve_run_name(tmp_path, TaskKind.OBB, "exp", auto_inc=False) == "exp"
