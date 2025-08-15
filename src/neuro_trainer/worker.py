# src/neuro_trainer/gui/worker.py
from __future__ import annotations
from typing import Callable, Optional
from PySide6.QtCore import QThread, Signal

class WorkerThread(QThread):
    started_ok = Signal()
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(self, target: Callable[[], None], parent=None) -> None:
        super().__init__(parent)
        self._target = target

    def run(self) -> None:
        try:
            self.started_ok.emit()
            self._target()
            self.finished_ok.emit()
        except Exception as e:
            self.failed.emit(str(e))
