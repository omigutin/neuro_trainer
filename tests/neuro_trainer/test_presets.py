import pytest

from neuro_trainer.presets import PresetRegistry
from neuro_trainer.state import Profile
from neuro_trainer.tasks import TaskKind

from ..conftest import ROOT


@pytest.mark.parametrize("task", list(TaskKind))
@pytest.mark.parametrize("profile", list(Profile))
def test_all_presets_load(task, profile):
    values = PresetRegistry(ROOT / "configs" / "presets").get(task, profile)
    assert values["epochs"] >= 1 and values["batch"] >= 1


def test_unknown_key_rejected(tmp_path):
    (tmp_path / "det_fast.yaml").write_text("epochs: 5\nlabel_smoothing: 0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="label_smoothing"):
        PresetRegistry(tmp_path).get(TaskKind.DETECTOR, Profile.FAST)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        PresetRegistry(tmp_path).get(TaskKind.OBB, Profile.FAST)
