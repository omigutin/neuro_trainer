# src/neuro_trainer/gui/state.py
from __future__ import annotations
from PySide6.QtCore import QSettings, QByteArray

ORG = "model_trainer"
APP = "neuro_trainer_gui"

class WindowState:
    def __init__(self) -> None:
        self._s = QSettings(ORG, APP)

    def save(self, geometry: QByteArray, state: QByteArray, current_tab: int) -> None:
        self._s.setValue("geometry", geometry)
        self._s.setValue("windowState", state)
        self._s.setValue("currentTab", current_tab)

    def restore_geometry(self):
        return self._s.value("geometry")

    def restore_state(self):
        return self._s.value("windowState")

    def restore_tab(self, default: int = 0) -> int:
        v = self._s.value("currentTab")
        try:
            return int(v)
        except Exception:
            return default
