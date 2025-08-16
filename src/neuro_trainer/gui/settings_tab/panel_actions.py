from __future__ import annotations
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QFileDialog, QMessageBox

class PanelActions(QWidget):
    load_clicked = Signal()
    save_clicked = Signal()
    validate_clicked = Signal()
    start_clicked = Signal()

    def __init__(self, app, parent=None, btn_width: int = 170):
        super().__init__(parent)
        self.app = app
        h = QHBoxLayout(self)
        h.setContentsMargins(6, 0, 6, 6)

        self.btn_load = QPushButton("Загрузить пресет"); self.btn_load.clicked.connect(self._load)
        self.btn_save = QPushButton("Сохранить пресет"); self.btn_save.clicked.connect(self._save)
        self.btn_validate = QPushButton("Валидация"); self.btn_validate.clicked.connect(self.validate_clicked.emit)
        self.btn_start = QPushButton("Старт"); self.btn_start.clicked.connect(self.start_clicked.emit)

        for b in (self.btn_load, self.btn_save, self.btn_validate, self.btn_start):
            b.setMinimumWidth(btn_width)

        h.addWidget(self.btn_load)
        h.addWidget(self.btn_save)
        h.addStretch(1)
        h.addWidget(self.btn_validate)
        h.addWidget(self.btn_start)

    def _load(self):
        base = self.app._paths.presets_user_dir.as_posix()
        file, _ = QFileDialog.getOpenFileName(self, "Загрузить пресет YAML", base, "YAML (*.yaml *.yml)")
        if not file: return
        try:
            self.app.load_preset_yaml(Path(file))
            self.load_clicked.emit()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка YAML", str(e))

    def _save(self):
        base = self.app._paths.presets_user_dir.as_posix()
        file, _ = QFileDialog.getSaveFileName(self, "Сохранить пресет YAML", base, "YAML (*.yaml *.yml)")
        if not file: return
        try:
            self.app.save_preset_yaml(Path(file))
            self.save_clicked.emit()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка YAML", str(e))
