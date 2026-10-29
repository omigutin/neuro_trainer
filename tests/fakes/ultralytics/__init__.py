""" Минимальная подмена ultralytics для тестов: тот же интерфейс YOLO, без torch. """

import csv
import os
import time
import types
from pathlib import Path

# Аргументы model.train(), которые принимает подмена (как в ultralytics/cfg/default.yaml 8.4)
_KEYS_FILE = Path(__file__).resolve().parent.parent / "ultralytics_8_4_cfg_keys.txt"
ALLOWED_TRAIN_ARGS = frozenset(
    line.strip() for line in _KEYS_FILE.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")
)
# Пауза на эпоху (для тестов остановки)
EPOCH_SLEEP = float(os.environ.get("FAKE_UL_EPOCH_SLEEP", "0"))

_TASK_BY_SUFFIX = (("-cls", "classify"), ("-seg", "segment"), ("-pose", "pose"), ("-obb", "obb"))


class YOLO:
    def __init__(self, weights):
        self.weights = str(weights)
        name = Path(self.weights).stem
        self.task = next((t for s, t in _TASK_BY_SUFFIX if name.endswith(s)), "detect")
        self.names = {0: "a", 1: "b"}
        self.ckpt = {"train_args": {"imgsz": 320}, "epoch": 3 if name == "last" else -1,
                     "optimizer": {} if name == "last" else None}
        self.ckpt_path = self.weights
        self.callbacks = {}
        self.train_kwargs = None

    def add_callback(self, event, func):
        self.callbacks.setdefault(event, []).append(func)

    def _run(self, event, trainer):
        for f in self.callbacks.get(event, []):
            f(trainer)

    def train(self, **kw):
        if ALLOWED_TRAIN_ARGS:
            bad = set(kw) - ALLOWED_TRAIN_ARGS - {"resume"}
            if bad:
                raise SyntaxError(f"not a valid YOLO argument: {sorted(bad)}")
        self.train_kwargs = kw
        if kw.get("resume"):
            save_dir = Path(self.weights).parent.parent
            start, epochs = 3, 5
        else:
            save_dir = Path(kw["project"]) / kw["name"]
            start, epochs = 0, int(kw.get("epochs", 3))
        save_dir.mkdir(parents=True, exist_ok=True)
        tr = types.SimpleNamespace(epoch=start, epochs=epochs, stop=False, metrics={}, lr={"lr/pg0": 0.01},
                                   tloss=[1.0, 2.0])
        tr.label_loss_items = lambda loss, prefix="train": {f"{prefix}/box_loss": loss[0], f"{prefix}/cls_loss": loss[1]}
        self._run("on_train_start", tr)
        rows = []
        for epoch in range(start, epochs):
            tr.epoch = epoch
            tr.tloss = [1.0 / (epoch + 1), 2.0 / (epoch + 1)]
            tr.metrics = {"metrics/mAP50-95(B)": 0.1 * (epoch + 1), "metrics/mAP50(B)": 0.2 * (epoch + 1),
                          "val/box_loss": 0.5 / (epoch + 1)}
            rows.append({"epoch": epoch + 1, **tr.metrics})
            (save_dir / "weights").mkdir(exist_ok=True)
            (save_dir / "weights" / "last.pt").write_text("w")
            self._run("on_fit_epoch_end", tr)
            if EPOCH_SLEEP:
                time.sleep(EPOCH_SLEEP)
            if tr.stop:
                break
        with (save_dir / "results.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[-1]))
            w.writeheader()
            w.writerows(rows)
        print("fake training done", flush=True)

    def val(self, **kw):
        save_dir = Path(kw["project"]) / kw["name"]
        save_dir.mkdir(parents=True, exist_ok=True)
        return types.SimpleNamespace(results_dict={"metrics/mAP50-95(B)": 0.42})
