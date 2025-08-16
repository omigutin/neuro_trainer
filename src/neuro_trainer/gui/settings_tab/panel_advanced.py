from __future__ import annotations
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGroupBox, QGridLayout, QLabel, QComboBox, QDoubleSpinBox
)

class PanelAdvanced(QGroupBox):
    state_changed = Signal()

    def __init__(self, parent=None, field_width: int = 260):
        super().__init__("Доп. настройки", parent)
        self.setCheckable(True)
        self.setChecked(True)
        self._field_w = field_width

        g = QGridLayout(self)
        g.setContentsMargins(12, 12, 12, 12)
        g.setHorizontalSpacing(28)    # существенный промежуток между столбцами
        g.setVerticalSpacing(8)

        def fspin(lo, hi, step, dv):
            w = QDoubleSpinBox()
            w.setRange(lo, hi)
            w.setSingleStep(step)
            w.setDecimals(dv)
            w.setFixedWidth(self._field_w)
            w.valueChanged.connect(self.state_changed)
            return w

        # список полей
        labels = [
            ("hsv_h", fspin(0.0, 1.0, 0.005, 3)),
            ("hsv_s", fspin(0.0, 1.0, 0.01, 3)),
            ("hsv_v", fspin(0.0, 1.0, 0.01, 3)),
            ("degrees", fspin(0.0, 45.0, 0.5, 2)),
            ("translate", fspin(0.0, 0.5, 0.01, 2)),
            ("scale", fspin(0.0, 2.0, 0.05, 2)),
            ("shear", fspin(0.0, 10.0, 0.5, 2)),
            ("perspective", fspin(0.0, 0.001, 0.0001, 4)),
            ("flipud", fspin(0.0, 1.0, 0.05, 2)),
            ("fliplr", fspin(0.0, 1.0, 0.05, 2)),
            ("mosaic", fspin(0.0, 1.0, 0.1, 2)),
            ("mixup", fspin(0.0, 1.0, 0.1, 2)),
            ("copy_paste (seg)", fspin(0.0, 1.0, 0.1, 2)),
        ]
        # CLS-специфика
        self.auto_augment = QComboBox()
        self.auto_augment.addItems(["randaugment", "augment", "none"])
        self.auto_augment.setFixedWidth(self._field_w)
        self.auto_augment.currentIndexChanged.connect(self.state_changed)
        labels += [
            ("auto_augment", self.auto_augment),
            ("erasing", fspin(0.0, 1.0, 0.05, 2)),
            ("label_smoothing", fspin(0.0, 0.2, 0.01, 2)),
            ("dropout", fspin(0.0, 0.9, 0.05, 2)),
        ]

        # разложим по 3 столбцам по 6 полей
        col = 0; row = 0
        self._widgets: dict[str, object] = {}
        for text, w in labels:
            g.addWidget(QLabel(text), row, col * 2)
            g.addWidget(w, row, col * 2 + 1)
            self._widgets[text] = w
            row += 1
            if row == 6:
                row = 0; col += 1

    # API
    def set_values(self, s):
        def setv(name: str, val):
            w = self._widgets[name]
            if hasattr(w, "setValue"):
                w.setValue(val)                 # QDoubleSpinBox
            elif hasattr(w, "setCurrentText"):
                w.setCurrentText(val)           # QComboBox
            else:
                raise AttributeError(f"Unsupported widget for {name}")

        setv("hsv_h", s.hsv_h); setv("hsv_s", s.hsv_s); setv("hsv_v", s.hsv_v)
        setv("degrees", s.degrees); setv("translate", s.translate); setv("scale", s.scale)
        setv("shear", s.shear); setv("perspective", s.perspective); setv("flipud", s.flipud); setv("fliplr", s.fliplr)
        setv("mosaic", s.mosaic); setv("mixup", s.mixup); setv("copy_paste (seg)", s.copy_paste)
        setv("auto_augment", s.auto_augment)
        setv("erasing", s.erasing); setv("label_smoothing", s.label_smoothing); setv("dropout", s.dropout)

    def get_values(self) -> dict:
        def getv(name: str):
            w = self._widgets[name]
            if hasattr(w, "value"):
                return w.value()                 # QDoubleSpinBox
            elif hasattr(w, "currentText"):
                return w.currentText()           # QComboBox
            else:
                raise AttributeError(f"Unsupported widget for {name}")

        return dict(
            auto_augment=getv("auto_augment"),
            hsv_h=float(getv("hsv_h")), hsv_s=float(getv("hsv_s")), hsv_v=float(getv("hsv_v")),
            degrees=float(getv("degrees")), translate=float(getv("translate")), scale=float(getv("scale")),
            shear=float(getv("shear")), perspective=float(getv("perspective")),
            flipud=float(getv("flipud")), fliplr=float(getv("fliplr")),
            mosaic=float(getv("mosaic")), mixup=float(getv("mixup")), copy_paste=float(getv("copy_paste (seg)")),
            erasing=float(getv("erasing")), label_smoothing=float(getv("label_smoothing")), dropout=float(getv("dropout")),
        )
