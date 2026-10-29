from pathlib import Path

from neuro_trainer.web.native_pickers import nearest_existing_dir


def test_nearest_existing_dir(tmp_path):
    assert nearest_existing_dir(str(tmp_path / "a" / "b.pt")) == tmp_path
    assert nearest_existing_dir(str(tmp_path)) == tmp_path
    assert nearest_existing_dir("") == Path.home()
