from __future__ import annotations
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGroupBox, QGridLayout, QLabel
from PySide6.QtGui import QCursor

LINKS = [
    ("Ultralytics — Главная", "https://docs.ultralytics.com", "Быстрый старт и общая документация"),
    ("YOLO: Классификация", "https://docs.ultralytics.com/tasks/classify/", "Руководство по классификации"),
    ("YOLO: Детекция", "https://docs.ultralytics.com/tasks/detect/", "Руководство по детекции"),
    ("YOLO: Сегментация", "https://docs.ultralytics.com/tasks/segment/", "Руководство по сегментации"),
    ("train()", "https://docs.ultralytics.com/modes/train/", "Параметры обучения"),
    ("val()", "https://docs.ultralytics.com/modes/val/", "Параметры валидации"),
    ("Артефакты/пути", "https://docs.ultralytics.com/guides/runs/", "Как YOLO раскладывает результаты"),
    ("Аугментации", "https://docs.ultralytics.com/usage/augmentation/", "Описание аугментаций"),
    ("Оптимизация (LR, Momentum…)", "https://docs.ultralytics.com/usage/hyperparameters/", "Гиперпараметры оптимизации"),
    ("Релизы YOLO", "https://github.com/ultralytics/ultralytics/releases", "История версий"),
    # Можно добавить хорошие статьи/ноутбуки
    ("Roboflow: практики", "https://blog.roboflow.com", "Практические статьи по датасетам/аугментациям"),
    ("Albumentations guide", "https://albumentations.ai/docs/", "Популярные аугментации изображений"),
]

class QuickLinks(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Быстрые ссылки", parent)
        g = QGridLayout(self); g.setContentsMargins(6,6,6,6); g.setHorizontalSpacing(18); g.setVerticalSpacing(4)

        def make_link(text:str, url:str, tip:str) -> QLabel:
            lbl = QLabel(f'<a href="{url}">{text}</a>')
            lbl.setTextFormat(Qt.RichText)
            lbl.setTextInteractionFlags(Qt.TextBrowserInteraction)
            lbl.setOpenExternalLinks(True)
            lbl.setToolTip(tip)
            lbl.setCursor(QCursor(Qt.PointingHandCursor))
            return lbl

        # 3 колонки
        cols = 3
        for i,(t,u,tip) in enumerate(LINKS):
            r, c = divmod(i, cols)
            g.addWidget(make_link(t,u,tip), r, c, Qt.AlignLeft)
