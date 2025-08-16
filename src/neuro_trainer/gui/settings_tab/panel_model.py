from __future__ import annotations
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QGroupBox, QGridLayout, QLabel, QComboBox, QLineEdit,
    QToolButton, QFileDialog, QStyle
)

from ...models_registry import ModelsRegistry
from ...weights_cache import WeightsCache


class PanelModel(QGroupBox):
    state_changed = Signal()
    task_auto_detected = Signal(str)

    def __init__(self, app, parent=None, field_width: int = 260):
        super().__init__("Модель и источник", parent)
        self.app = app
        self._field_w = field_width

        g = QGridLayout(self)
        g.setContentsMargins(12, 12, 12, 12)
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(8)

        # ряд 0: Тип / Версия модели / Модель (все одинаковой ширины)
        g.addWidget(QLabel("Тип"), 0, 0)
        self.cmb_task = QComboBox(); self.cmb_task.addItems(["auto", "detector", "classifier", "segmentation"])
        self.cmb_task.setFixedWidth(self._field_w)
        self.cmb_task.currentIndexChanged.connect(self._on_task_changed)
        g.addWidget(self.cmb_task, 0, 1)

        g.addWidget(QLabel("Версия модели"), 0, 2)
        self.cmb_version = QComboBox(); self.cmb_version.addItems(["YOLO_v8", "YOLO_v9", "YOLO_v10", "YOLO_v11"])
        self.cmb_version.setFixedWidth(self._field_w)
        self.cmb_version.currentIndexChanged.connect(self._on_version_changed)
        g.addWidget(self.cmb_version, 0, 3)

        g.addWidget(QLabel("Модель"), 0, 4)
        self.cmb_model = QComboBox()
        self.cmb_model.setFixedWidth(self._field_w)
        self.cmb_model.currentIndexChanged.connect(self._on_model_selected)
        g.addWidget(self.cmb_model, 0, 5)

        # ряд 1: Путь к dataset + кнопка папки
        g.addWidget(QLabel("Путь к dataset"), 1, 0)
        self.le_data = QLineEdit(); self.le_data.setPlaceholderText(r"папка датасета / data.yaml / Roboflow URL")
        self.le_data.textChanged.connect(self.state_changed)
        g.addWidget(self.le_data, 1, 1, 1, 5)

        self.btn_browse = QToolButton()
        self.btn_browse.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.btn_browse.clicked.connect(self._browse_dataset)
        g.addWidget(self.btn_browse, 1, 6)

        self._rebuild_models_list()

        # подсказки
        self.cmb_task.setToolTip("Тип задачи. 'auto' — определяется по выбранной модели.")
        self.cmb_version.setToolTip("Линейка моделей YOLO.")
        self.cmb_model.setToolTip("Сначала стандартные веса, затем — пользовательские (после разделителя). "
                                  "Кэшированные выделены зелёным.")
        self.btn_browse.setToolTip("Выбрать папку датасета")

    # --- API для MainWindow ---

    def get_task(self) -> str:
        return self.cmb_task.currentText()

    def set_task(self, val: str):
        self.cmb_task.setCurrentText(val)

    def get_model_display(self) -> str:
        return self.cmb_model.currentText()

    def get_model_name_for_state(self) -> str:
        raw = self.cmb_model.currentText().strip()
        # если это разделитель — считаем, что ничего не выбрано
        return "" if raw.startswith("---") or not raw else raw.split()[0]

    def set_model_name(self, name: str):
        for i in range(self.cmb_model.count()):
            if self.cmb_model.itemText(i).startswith(name):
                self.cmb_model.setCurrentIndex(i)
                break

    def get_data_path(self) -> str:
        return self.le_data.text().strip()

    def set_data_path(self, val: str):
        self.le_data.setText(val)

    def refresh_models(self):
        self._rebuild_models_list()

    # --- internal ---

    def _current_version_prefix(self) -> str:
        # YOLO_v11 -> "v11"
        return self.cmb_version.currentText().split("_")[-1].lower()

    def _std_models_filtered(self) -> list[str]:
        prefix = self._current_version_prefix()
        all_std = ModelsRegistry.get_all_models()
        all_std = [m for m in all_std if ("v" in m and prefix in m)]
        t = self.get_task()
        if t != "auto":
            all_std = [m for m in all_std if _matches_task(m, t)]
        return all_std

    def _rebuild_models_list(self):
        cache = WeightsCache(self.app._paths.weights_dir)
        cached = set(cache.list_cached())

        std = self._std_models_filtered()
        custom = sorted([m for m in cached if m not in ModelsRegistry.get_all_models()])

        self.cmb_model.blockSignals(True)
        self.cmb_model.clear()

        # стандартные (кэшированные зелёным)
        for m in std:
            self.cmb_model.addItem(m)
            idx = self.cmb_model.count() - 1
            if m in cached:
                self.cmb_model.model().item(idx).setForeground(QColor("#2E7D32"))  # зелёный

        # разделитель
        if custom:
            self.cmb_model.addItem("--- пользовательские ---")
            self.cmb_model.model().item(self.cmb_model.count() - 1).setEnabled(False)

            for m in custom:
                self.cmb_model.addItem(m)
                self.cmb_model.model().item(self.cmb_model.count() - 1).setForeground(QColor("#2E7D32"))

        self.cmb_model.blockSignals(False)

    def _on_task_changed(self):
        self._rebuild_models_list()
        self.state_changed.emit()

    def _on_version_changed(self):
        self._rebuild_models_list()
        self.state_changed.emit()

    def _on_model_selected(self):
        if self.get_task() != "auto":
            self.state_changed.emit()
            return
        txt = self.get_model_display().lower()
        if txt.startswith("---") or not txt:
            self.state_changed.emit()
            return
        if "cls" in txt:
            self.set_task("classifier"); self.task_auto_detected.emit("classifier")
        elif "seg" in txt:
            self.set_task("segmentation"); self.task_auto_detected.emit("segmentation")
        else:
            self.set_task("detector"); self.task_auto_detected.emit("detector")
        self.state_changed.emit()

    def _browse_dataset(self):
        base = str(Path(self.app._paths.default_runs_dir).parent)
        path = QFileDialog.getExistingDirectory(self, "Выберите папку датасета", base)
        if path:
            self.le_data.setText(path)
        self.state_changed.emit()


def _matches_task(name: str, task: str) -> bool:
    n = name.lower()
    if task == "classifier":
        return "cls" in n
    if task == "segmentation":
        return "seg" in n
    return ("cls" not in n) and ("seg" not in n)
