from __future__ import annotations
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QGroupBox, QGridLayout, QLabel, QComboBox, QSpinBox, QCheckBox, QLineEdit, QToolButton, QFileDialog
)

W_FIELD = 120
MARGIN = 6

class PanelResearch(QGroupBox):
    """Исследование и результаты: устройство/workers/deterministic + путь к result-папке."""
    state_changed = Signal()

    def __init__(self, app, parent=None):
        super().__init__("Исследование и результаты", parent)
        self.app = app

        g = QGridLayout(self)
        g.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(6)

        # первая строка — устройство / workers / deterministic
        g.addWidget(QLabel("Устройство"), 0, 0, Qt.AlignLeft)
        self.cmb_device = QComboBox(); self.cmb_device.addItems(["auto", "cuda", "mps", "cpu"])
        self.cmb_device.setFixedWidth(W_FIELD); self.cmb_device.currentIndexChanged.connect(self.state_changed)
        g.addWidget(self.cmb_device, 0, 1)

        g.addWidget(QLabel("Workers"), 0, 2, Qt.AlignLeft)
        self.sb_workers = QSpinBox(); self.sb_workers.setRange(0, 64); self.sb_workers.setFixedWidth(W_FIELD)
        self.sb_workers.valueChanged.connect(self.state_changed)
        g.addWidget(self.sb_workers, 0, 3)

        self.chk_det = QCheckBox("Deterministic"); self.chk_det.stateChanged.connect(self.state_changed)
        g.addWidget(self.chk_det, 0, 4)

        # строка 2 — путь к результатам
        g.addWidget(QLabel("Результаты"), 1, 0, Qt.AlignLeft)
        self.le_results = QLineEdit()
        self.le_results.setPlaceholderText("")
        self.le_results.textChanged.connect(self.state_changed)
        g.addWidget(self.le_results, 1, 1, 1, 4)

        btn = QToolButton(self); btn.setIcon(QIcon.fromTheme("folder"))
        btn.setToolTip("Выбрать папку для результатов")
        btn.clicked.connect(self._browse_results)
        g.addWidget(btn, 1, 5)

    # API
    def set_values(self, device: str, workers: int, deterministic: bool, results_dir: str):
        self.cmb_device.setCurrentText(device or "auto")
        self.sb_workers.setValue(int(workers))
        self.chk_det.setChecked(bool(deterministic))
        self.le_results.setText(results_dir or "")

    def get_results_dir(self) -> str:
        return self.le_results.text().strip()

    def get_device_pack(self) -> tuple[str, int, bool]:
        return (self.cmb_device.currentText(), int(self.sb_workers.value()), bool(self.chk_det.isChecked()))

    # internal
    def _browse_results(self):
        base = self.le_results.text().strip() or str(Path(self.app._paths.default_runs_dir))
        path = QFileDialog.getExistingDirectory(self, "Папка результатов", base)
        if path:
            self.le_results.setText(path)
        self.state_changed.emit()
