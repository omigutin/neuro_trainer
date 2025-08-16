from __future__ import annotations

from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QLabel
from PySide6.QtCore import Qt

_LINKS = [
    ("Ultralytics — Главная",       "https://docs.ultralytics.com",                 "Главная страница документации."),
    ("YOLO: Классификация",         "https://docs.ultralytics.com/tasks/classify/", "Раздел по классификации."),
    ("YOLO: Детекция",              "https://docs.ultralytics.com/tasks/detect/",   "Раздел по детекции."),
    ("YOLO: Сегментация",           "https://docs.ultralytics.com/tasks/segment/",  "Раздел по сегментации."),
    ("Режим train()",               "https://docs.ultralytics.com/modes/train/",    "Все параметры обучения."),
    ("Режим val()",                 "https://docs.ultralytics.com/modes/val/",      "Все параметры валидации."),
]

class PanelLinks(QGroupBox):
    """Панель с «кликабельными» ссылками (как ссылки, не кнопки)."""
    def __init__(self, parent=None):
        super().__init__("Быстрые ссылки")
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(4)

        for text, url, tip in _LINKS:
            lbl = QLabel(f'<a href="{url}">{text}</a>')
            lbl.setOpenExternalLinks(True)
            lbl.setToolTip(tip)
            lbl.setTextInteractionFlags(Qt.TextBrowserInteraction)
            v.addWidget(lbl)
