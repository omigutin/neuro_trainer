from __future__ import annotations
from PySide6.QtGui import QIntValidator, QValidator
from PySide6.QtWidgets import QLineEdit

class BoundedIntValidator(QIntValidator):
    """Валидатор целых значений с границами (разрешает пустую строку как Intermediate)."""
    def __init__(self, lo: int, hi: int, parent=None):
        super().__init__(lo, hi, parent)

    # PySide6 ожидает кортеж (QValidator.State, str, int)
    def validate(self, input_str: str, pos: int):
        if input_str.strip() == "":
            return (QValidator.Intermediate, input_str, pos)
        return super().validate(input_str, pos)

def valid_imgsz_string(s: str) -> bool:
    s = s.strip().lower()
    if not s:
        return False
    if "x" in s:
        try:
            a, b = s.split("x", 1)
            ai, bi = int(a), int(b)
            return 64 <= ai <= 2048 and 64 <= bi <= 2048
        except Exception:
            return False
    try:
        v = int(s)
        return 64 <= v <= 2048
    except Exception:
        return False

class ImgSizeLineEdit(QLineEdit):
    """Поле ввода для imgsz: принимает int или 'HxW'; при потере фокуса округляет до кратности 32."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPlaceholderText("CLS: 384  |  DET/SEG: 640 или 640x960")

    def focusOutEvent(self, e):
        txt = self.text().strip().lower()
        if txt:
            try:
                if "x" in txt:
                    a, b = txt.split("x", 1)
                    ai, bi = int(a), int(b)
                    ai = self._round32(ai); bi = self._round32(bi)
                    self.setText(f"{ai}x{bi}")
                else:
                    v = int(txt)
                    self.setText(str(self._round32(v)))
            except Exception:
                pass
        return super().focusOutEvent(e)

    @staticmethod
    def _round32(v: int) -> int:
        return int(round(v / 32) * 32)
