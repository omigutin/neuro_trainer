from __future__ import annotations
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter, QListWidget, QListWidgetItem, QTextBrowser
)
from .quick_links import QuickLinks
from ...help_content import HelpContent

SECTIONS_ORDER = [
    ("intro", "Как быстро запустить обучение"),
    ("classifier", "Классификация (Classifier)"),
    ("detector", "Детекция (Detector)"),
    ("segmentator", "Сегментация (Segmentator)"),
    ("train_params", "Параметры train() — простым языком"),
    ("val_params", "Параметры val() — простым языком"),
]

class PanelHelp(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self); v.setContentsMargins(6,6,6,6); v.setSpacing(8)

        v.addWidget(QuickLinks(self))

        split = QSplitter(Qt.Horizontal, self)
        self.toc = QListWidget(); self.toc.setMinimumWidth(260)
        self.doc = QTextBrowser(); self.doc.setOpenExternalLinks(True)

        self._build_doc()

        for key, title in SECTIONS_ORDER:
            item = QListWidgetItem(title); item.setData(Qt.UserRole, key)
            self.toc.addItem(item)

        self.toc.currentRowChanged.connect(self._on_select)
        self.toc.setCurrentRow(0)

        split.addWidget(self.toc); split.addWidget(self.doc)
        split.setStretchFactor(0, 0); split.setStretchFactor(1, 1)
        v.addWidget(split, 1)

    def _build_doc(self):
        s = HelpContent.get_sections()
        # соберем единый HTML с якорями
        html = []
        html.append(f'<h2 id="intro">{SECTIONS_ORDER[0][1]}</h2>')
        html.append(HelpContent._md_to_html(s["intro"]))  # приватный конвертер см. обновлённый HelpContent
        html.append(f'<h2 id="classifier">{SECTIONS_ORDER[1][1]}</h2>')
        html.append(HelpContent._md_to_html(s["classifier"]))
        html.append(f'<h2 id="detector">{SECTIONS_ORDER[2][1]}</h2>')
        html.append(HelpContent._md_to_html(s["detector"]))
        html.append(f'<h2 id="segmentator">{SECTIONS_ORDER[3][1]}</h2>')
        html.append(HelpContent._md_to_html(s["segmentator"]))
        html.append(f'<h2 id="train_params">{SECTIONS_ORDER[4][1]}</h2>')
        html.append(HelpContent._md_to_html(s["train_params"]))
        html.append(f'<h2 id="val_params">{SECTIONS_ORDER[5][1]}</h2>')
        html.append(HelpContent._md_to_html(s["val_params"]))
        self.doc.setHtml("\n".join(html))

    def _on_select(self, row: int):
        key = self.toc.item(row).data(Qt.UserRole)
        self.doc.scrollToAnchor(key)
