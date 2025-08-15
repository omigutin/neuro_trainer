from __future__ import annotations
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import QLineEdit
from PySide6.QtCore import Qt

class BoundedIntValidator(QIntValidator):
    """Целое число в [lo..hi]."""
    def __init__(self, lo: int, hi: int, parent=None):
        super().__init__(lo, hi, parent)

class ImgSizeLineEdit(QLineEdit):
    """
    Поле для imgsz: принимает
      - одно число (например, 384)
      - строку 'HxW' (например, '640x960')
    При потере фокуса приводит к кратному 32:
      - CLS: квадратное число (int)
      - DET/SEG: каждую сторону кратной 32
    Здесь не знаем тип задачи, поэтому округляем обе стороны; CLS добьём на стороне App.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPlaceholderText("384 или 640x960")

    def focusOutEvent(self, e):
        text = self.text().strip().lower()
        rounded = self._round_to_stride(text)
        if rounded:
            self.setText(rounded)
        super().focusOutEvent(e)

    @staticmethod
    def _round_to_stride(s: str, stride: int = 32) -> str | None:
        def round32(v: int) -> int:
            return max(32, round(v / stride) * stride)
        if not s:
            return None
        try:
            if "x" in s:
                a, b = s.split("x", 1)
                ha, hb = round32(int(a)), round32(int(b))
                return f"{int(ha)}x{int(hb)}"
            else:
                return str(int(round32(int(s))))
        except Exception:
            return None

    def value(self) -> int | tuple[int, int] | None:
        s = self.text().strip().lower()
        if not s:
            return None
        try:
            if "x" in s:
                a, b = s.split("x", 1)
                return (int(a), int(b))
            return int(s)
        except Exception:
            return None

def valid_imgsz_string(s: str) -> bool:
    s = s.strip().lower()
    if not s:
        return False
    try:
        if "x" in s:
            a, b = s.split("x", 1)
            ai, bi = int(a), int(b)
            return 64 <= ai <= 2048 and 64 <= bi <= 2048
        else:
            ai = int(s)
            return 64 <= ai <= 2048
    except Exception:
        return False
