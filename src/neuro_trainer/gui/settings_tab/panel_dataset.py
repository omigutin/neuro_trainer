from __future__ import annotations
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGroupBox, QGridLayout, QLabel, QComboBox, QLineEdit, QSpinBox,
    QCheckBox, QToolButton, QFileDialog, QStyle
)

class PanelDataset(QGroupBox):
    state_changed = Signal()

    def __init__(self, app, parent=None, field_width: int = 260):
        super().__init__("Исследование и результаты", parent)
        self.app = app
        self._field_w = field_width

        g = QGridLayout(self)
        g.setContentsMargins(12, 12, 12, 12)
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(8)

        # строка 0 — устройство / workers / deterministic
        g.addWidget(QLabel("Устройство"), 0, 0)
        self.cmb_device = QComboBox(); self.cmb_device.addItems(["auto", "cuda", "mps", "cpu"])
        self.cmb_device.setFixedWidth(self._field_w)
        self.cmb_device.currentIndexChanged.connect(self.state_changed)
        g.addWidget(self.cmb_device, 0, 1)

        g.addWidget(QLabel("Workers"), 0, 2)
        self.sb_workers = QSpinBox(); self.sb_workers.setRange(0, 64)
        self.sb_workers.setFixedWidth(self._field_w)
        self.sb_workers.valueChanged.connect(self.state_changed)
        g.addWidget(self.sb_workers, 0, 3)

        self.chk_det = QCheckBox("Deterministic")
        self.chk_det.stateChanged.connect(self.state_changed)
        g.addWidget(self.chk_det, 0, 4)

        # строка 1 — Результаты + папка + Название
        g.addWidget(QLabel("Результаты"), 1, 0)
        self.le_results = QLineEdit()
        g.addWidget(self.le_results, 1, 1, 1, 3)

        self.btn_results = QToolButton()
        self.btn_results.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.btn_results.setToolTip("Выбрать папку для сохранения результатов")
        self.btn_results.clicked.connect(self._browse_results)
        g.addWidget(self.btn_results, 1, 4)

        g.addWidget(QLabel("Название"), 1, 5)
        self.le_runname = QLineEdit(); self.le_runname.setPlaceholderText("runN")
        self.le_runname.setFixedWidth(self._field_w)
        self.le_runname.textChanged.connect(self.state_changed)
        g.addWidget(self.le_runname, 1, 6)

        # строка 2 — (оставлена пустой под расширение при необходимости)

    # API
    def set_values(self, *, device: str, workers: int, deterministic: bool, results_dir: str, run_name: str):
        self.cmb_device.setCurrentText(device or "auto")
        self.sb_workers.setValue(int(workers))
        self.chk_det.setChecked(bool(deterministic))
        self.le_results.setText(results_dir or "")
        self.le_runname.setText(run_name or "")

    def get_results_dir(self) -> str:
        return self.le_results.text().strip()

    def get_run_name(self) -> str:
        return self.le_runname.text().strip()

    # browse
    def _browse_results(self):
        base = str(Path(self.app._paths.default_runs_dir).parent)
        path = QFileDialog.getExistingDirectory(self, "Выберите папку результатов", base)
        if path:
            self.le_results.setText(path)
        self.state_changed.emit()

    # прежние методы совместимости (часть полей была здесь раньше)
    def get_data(self) -> str:
        return ""  # не используется (путь к датасету теперь в PanelModel)
    def set_data(self, path: str):
        pass
