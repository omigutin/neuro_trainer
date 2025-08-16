from __future__ import annotations
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGroupBox, QGridLayout, QLabel, QLineEdit, QCheckBox
)
from neuro_trainer.gui.settings_tab.validators import BoundedIntValidator, ImgSizeLineEdit, valid_imgsz_string

class PanelParams(QGroupBox):
    state_changed = Signal()

    def __init__(self, parent=None, field_width: int = 260):
        super().__init__("Гиперпараметры", parent)
        self._field_w = field_width

        g = QGridLayout(self)
        g.setContentsMargins(12, 12, 12, 12)
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(8)

        self.le_epochs = QLineEdit();   self.le_epochs.setValidator(BoundedIntValidator(1, 1000)); self.le_epochs.setFixedWidth(self._field_w)
        self.le_patience = QLineEdit(); self.le_patience.setValidator(BoundedIntValidator(0, 300));  self.le_patience.setFixedWidth(self._field_w)
        self.le_batch = QLineEdit();    self.le_batch.setValidator(BoundedIntValidator(1, 1024));  self.le_batch.setFixedWidth(self._field_w)
        self.le_imgsz = ImgSizeLineEdit(); self.le_imgsz.setFixedWidth(self._field_w)

        g.addWidget(QLabel("epochs"), 0, 0); g.addWidget(self.le_epochs, 0, 1)
        g.addWidget(QLabel("batch"), 0, 2); g.addWidget(self.le_batch, 0, 3)
        g.addWidget(QLabel("patience"), 1, 0); g.addWidget(self.le_patience, 1, 1)
        g.addWidget(QLabel("imgsz"), 1, 2); g.addWidget(self.le_imgsz, 1, 3)

        self.cb_rect = QCheckBox("rect (только DET/SEG)"); self.cb_rect.setToolTip("Упорядоченные по соотношению сторон батчи.")
        self.cb_multiscale = QCheckBox("multi_scale (только DET/SEG)"); self.cb_multiscale.setToolTip("Случайный масштаб на лету.")
        g.addWidget(self.cb_rect, 0, 4)
        g.addWidget(self.cb_multiscale, 1, 4)

        for w in (self.le_epochs, self.le_patience, self.le_batch, self.le_imgsz):
            w.textChanged.connect(self.state_changed)
        self.cb_rect.stateChanged.connect(self.state_changed)
        self.cb_multiscale.stateChanged.connect(self.state_changed)

    # API
    def set_values(self, epochs:int, patience:int, batch:int, imgsz_val:str, rect:bool, multiscale:bool):
        self.le_epochs.setText(str(epochs)); self.le_patience.setText(str(patience))
        self.le_batch.setText(str(batch)); self.le_imgsz.setText(imgsz_val)
        self.cb_rect.setChecked(rect); self.cb_multiscale.setChecked(multiscale)

    def get_values(self):
        return dict(
            epochs=int(self.le_epochs.text() or "0"),
            patience=int(self.le_patience.text() or "0"),
            batch=int(self.le_batch.text() or "0"),
            imgsz=self.le_imgsz.text().strip(),
            rect=self.cb_rect.isChecked(),
            multi_scale=self.cb_multiscale.isChecked(),
        )

    def enable_rect_opts(self, enabled: bool):
        self.cb_rect.setEnabled(enabled)
        self.cb_multiscale.setEnabled(enabled)

    def is_valid(self) -> bool:
        ok = (self.le_epochs.hasAcceptableInput()
              and self.le_patience.hasAcceptableInput()
              and self.le_batch.hasAcceptableInput()
              and valid_imgsz_string(self.le_imgsz.text()))
        return ok
