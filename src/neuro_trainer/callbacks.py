from __future__ import annotations
"""
Колбэки и флаг «мягкой» остановки обучения.
Trainer сейчас пишет JSONL сам через свои callbacks, но эти утилиты могут пригодиться
для дополнительных сценариев (условный ранний стоп, внешний лог).
"""

from pathlib import Path
from typing import Any
import json
from datetime import datetime


# ---------- STOP FLAG ----------
class StopFlag:
    """
    Запрос «мягкой» остановки через файл в папке рана.
    Работает между процессами: веб-сервер создаёт файл, процесс обучения проверяет его после каждой эпохи.
    """

    def __init__(self, path: Path) -> None:
        # Путь к файлу-флагу
        self.path = Path(path)

    def request(self) -> None:
        """ Попросить остановиться после текущей эпохи. """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")

    def reset(self) -> None:
        """ Снять запрос остановки. """
        self.path.unlink(missing_ok=True)

    def is_set(self) -> bool:
        """ Есть ли запрос остановки. """
        return self.path.exists()


# ---------- EARLY STOPS ----------
class EarlyStopOnTop1:
    def __init__(self, threshold: float = 0.95, patience: int = 3) -> None:
        self.t = threshold
        self.p = patience
        self.streak = 0

    def __call__(self, trainer: Any) -> None:
        m = getattr(trainer, "metrics", None)
        if m is None:
            return
        top1 = float(getattr(m, "top1", 0.0))
        self.streak = self.streak + 1 if top1 >= self.t else 0
        if self.streak >= self.p:
            print(f"[Callback] Early stop: top1 ≥ {self.t} for {self.p} epochs")
            trainer.stop = True


class EarlyStopOnMap:
    def __init__(self, threshold: float = 0.55, patience: int = 5) -> None:
        self.t = threshold
        self.p = patience
        self.streak = 0

    def __call__(self, trainer: Any) -> None:
        m = getattr(trainer, "metrics", None)
        if m is None:
            return
        box = getattr(m, "box", None)
        current = float(getattr(box, "map", 0.0)) if box is not None else 0.0
        self.streak = self.streak + 1 if current >= self.t else 0
        if self.streak >= self.p:
            print(f"[Callback] Early stop: mAP50-95 ≥ {self.t} for {self.p} epochs")
            trainer.stop = True


# ---------- JSONL LOGGER (опционально) ----------
class JsonlLogger:
    """
    Пишет JSON-строку после каждой эпохи:
      {"time":"...","task":"classify","epoch":7,"epochs":80,"top1":0.91,"top5":0.99}
      {"time":"...","task":"detect","epoch":12,"epochs":150,"map50_95":0.52,"map50":0.74,"map75":0.41}
    Если Trainer уже пишет JSONL — второй логгер не обязателен.
    """
    def __init__(self, log_path: Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, trainer: Any) -> None:
        m = getattr(trainer, "metrics", None)
        args = getattr(trainer, "args", None)
        task = getattr(args, "task", "") or ""
        epoch = int(getattr(trainer, "epoch", -1)) + 1  # Ultralytics хранит 0-based
        epochs = int(getattr(trainer, "epochs", 0))
        row: dict[str, float | int | str] = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "task": task, "epoch": epoch, "epochs": epochs,
        }
        try:
            if task == "classify":
                row.update(top1=float(getattr(m, "top1", 0.0)),
                           top5=float(getattr(m, "top5", 0.0)))
            else:
                box = getattr(m, "box", None)
                if box is not None:
                    row.update(map50_95=float(getattr(box, "map", 0.0)),
                               map50=float(getattr(box, "map50", 0.0)),
                               map75=float(getattr(box, "map75", 0.0)))
        except Exception:
            pass
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
