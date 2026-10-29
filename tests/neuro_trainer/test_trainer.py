import json

from neuro_trainer.trainer import PROGRESS_FILE, Trainer, epoch_row
from neuro_trainer.tasks import TaskKind


def _trainer(tmp_path, **kw):
    args = dict(weights="yolo26n.pt", task=TaskKind.DETECTOR, data="d.yaml", project_dir=str(tmp_path),
                run_name="run1", train_args={"epochs": 3, "batch": 2})
    args.update(kw)
    return Trainer(**args)


def test_train_writes_progress_into_run_dir(tmp_path, fake_ultralytics):
    t = _trainer(tmp_path)
    t.train()
    assert t.save_dir == tmp_path / "detect" / "run1"
    rows = [json.loads(x) for x in (t.save_dir / PROGRESS_FILE).read_text().splitlines()]
    assert [r["epoch"] for r in rows] == [1, 2, 3]
    assert rows[0]["metrics/mAP50-95(B)"] == 0.1 and rows[0]["train/box_loss"] == 1.0 and "lr/pg0" in rows[0]
    assert (t.save_dir / "final_metrics.json").is_file()


def test_train_kwargs(tmp_path):
    kw = _trainer(tmp_path)._build_train_kwargs()
    assert kw["project"] == str((tmp_path / "detect").resolve()) and kw["name"] == "run1"
    assert kw["exist_ok"] is True and kw["epochs"] == 3 and kw["data"] == "d.yaml"


def test_stop_flag_stops_after_current_epoch(tmp_path, fake_ultralytics):
    t = _trainer(tmp_path, train_args={"epochs": 10})
    t.save_dir.mkdir(parents=True)
    import ultralytics

    orig = ultralytics.YOLO.add_callback

    def add_callback(self, event, func):
        if event == "on_fit_epoch_end":
            def wrapped(tr):
                if tr.epoch == 1:
                    t.stop_flag.request()
                func(tr)
            return orig(self, event, wrapped)
        return orig(self, event, func)

    ultralytics.YOLO.add_callback = add_callback
    try:
        t.train()
    finally:
        ultralytics.YOLO.add_callback = orig
    rows = (t.save_dir / PROGRESS_FILE).read_text().splitlines()
    assert len(rows) == 2


def test_resume_appends_to_original_run(tmp_path, fake_ultralytics):
    run = tmp_path / "detect" / "run7"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "last.pt").write_text("w")
    (run / PROGRESS_FILE).write_text('{"epoch": 1}\n{"epoch": 2}\n{"epoch": 3}\n')
    t = _trainer(tmp_path, weights=str(run / "weights" / "last.pt"), run_name="", resume=True)
    assert t.save_dir == run.resolve()
    assert t._build_train_kwargs() == {"resume": True}
    t.train()
    epochs = [json.loads(x)["epoch"] for x in (run / PROGRESS_FILE).read_text().splitlines()]
    assert epochs == [1, 2, 3, 4, 5]


def test_validate_only(tmp_path, fake_ultralytics):
    t = _trainer(tmp_path, train_args={"imgsz": 640, "epochs": 3, "device": "cpu"})
    assert t.validate_only() == {"metrics/mAP50-95(B)": 0.42}
    assert json.loads((t.save_dir / "val_metrics.json").read_text())["metrics/mAP50-95(B)"] == 0.42


class _Tensor:
    def __init__(self, v):
        self.v = v

    def item(self):
        return self.v


def test_epoch_row_skips_nan_and_converts_tensors():
    import types
    tr = types.SimpleNamespace(epoch=0, epochs=5, tloss=None, metrics={"a": _Tensor(0.5), "b": float("nan"), "c": "x"},
                               lr={"lr/pg0": 0.01})
    row = epoch_row(tr)
    assert row["a"] == 0.5 and "b" not in row and "c" not in row and row["epoch"] == 1
