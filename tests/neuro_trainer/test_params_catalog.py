import pytest

from neuro_trainer.params_catalog import (
    PARAMS, ParamKind, coerce_values, default_values, get_spec, parse_imgsz, train_kwargs,
)
from neuro_trainer.tasks import TaskKind

from ..conftest import REAL_ULTRALYTICS


def test_every_param_has_help_and_label():
    for p in PARAMS:
        assert p.label and len(p.help) > 20, p.key
        if p.kind is ParamKind.CHOICE:
            assert p.default in p.choices, p.key


def test_defaults_pass_own_validation():
    values, errors = coerce_values(default_values())
    assert errors == {}
    assert values["epochs"] == 100


@pytest.mark.parametrize("key,raw,expected", [
    ("epochs", "50", 50), ("epochs", 50.0, 50), ("rect", "true", True), ("rect", 0, False),
    ("lr0", "0.001", 0.001), ("imgsz", "480x640", (480, 640)), ("imgsz", 320, 320), ("optimizer", "AdamW", "AdamW"),
])
def test_coerce_ok(key, raw, expected):
    assert get_spec(key).coerce(raw) == expected


@pytest.mark.parametrize("key,raw", [
    ("epochs", 0), ("epochs", "1.5"), ("epochs", True), ("lr0", "abc"), ("optimizer", "sgd2"),
    ("imgsz", "10"), ("fliplr", 1.5), ("rect", "maybe"),
])
def test_coerce_errors(key, raw):
    with pytest.raises(ValueError):
        get_spec(key).coerce(raw)


def test_coerce_values_collects_errors_and_skips_unknown():
    values, errors = coerce_values({"epochs": 0, "batch": 8, "unknown": 1})
    assert values == {"batch": 8}
    assert set(errors) == {"epochs"}


def test_train_kwargs_filters_by_task_and_converts():
    values = default_values()
    values.update(device="auto", cache="false", imgsz=(480, 640), time=0.0, freeze=0)
    det = train_kwargs(values, TaskKind.DETECTOR)
    assert "device" not in det and det["cache"] is False and det["imgsz"] == 640
    assert "time" not in det and "freeze" not in det
    assert "copy_paste" not in det and "dropout" not in det and "auto_augment" not in det
    seg = train_kwargs(values, TaskKind.SEGMENTATION)
    assert "copy_paste" in seg and "mask_ratio" in seg
    cls = train_kwargs(values, TaskKind.CLASSIFIER)
    assert "dropout" in cls and "mosaic" not in cls and "box" not in cls
    assert train_kwargs({**values, "device": "cuda", "cache": "ram"}, TaskKind.POSE)["device"] == "cuda"


def test_parse_imgsz_list():
    assert parse_imgsz([480, 640]) == (480, 640)


@pytest.mark.skipif(not REAL_ULTRALYTICS, reason="ultralytics is not installed")
def test_catalog_keys_exist_in_ultralytics():
    from ultralytics.cfg import DEFAULT_CFG_DICT
    missing = [p.key for p in PARAMS if p.key not in DEFAULT_CFG_DICT]
    assert missing == []
