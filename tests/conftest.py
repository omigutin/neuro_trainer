import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
FAKES = Path(__file__).resolve().parent / "fakes"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _real_ultralytics_available() -> bool:
    try:
        import importlib.util
        spec = importlib.util.find_spec("ultralytics")
        return spec is not None and FAKES not in Path(spec.origin).parents
    except (ImportError, ValueError):
        return False


REAL_ULTRALYTICS = _real_ultralytics_available()


@pytest.fixture
def fake_ultralytics(monkeypatch):
    """ Подменить ultralytics в текущем процессе и в процессах обучения (через PYTHONPATH). """
    for name in [m for m in sys.modules if m == "ultralytics" or m.startswith("ultralytics.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(FAKES))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(FAKES), os.environ.get("PYTHONPATH", "")]))
    yield
    for name in [m for m in sys.modules if m == "ultralytics" or m.startswith("ultralytics.")]:
        sys.modules.pop(name, None)


@pytest.fixture
def paths(tmp_path):
    """ AppConfigPaths во временном каталоге с копией стандартных профилей. """
    import shutil
    from neuro_trainer.utils import AppConfigPaths
    root = tmp_path / "project"
    shutil.copytree(ROOT / "configs" / "presets", root / "configs" / "presets")
    return AppConfigPaths(project_root=root)
