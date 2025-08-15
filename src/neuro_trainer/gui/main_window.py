# src/neuro_trainer/gui/main_window.py  (ПОЛНАЯ ВЕРСИЯ С УЧЁТОМ РАБОТНИКА/STOP/DEVICE)
from __future__ import annotations
from pathlib import Path
import json

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox, QCheckBox, QGroupBox,
    QMessageBox, QRadioButton, QTabWidget, QTextEdit, QProgressBar
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ..neuro_trainer import NeuroTrainerApp
from ..config_store import ConfigStore
from ..models_registry import ModelsRegistry
from ..presets import PresetRegistry
from ..utils import AppConfigPaths, open_in_explorer
from .validators import BoundedIntValidator, ImgSizeLineEdit, valid_imgsz_string
from .help_dialog import HelpDialog
from .worker import WorkerThread
from .state import WindowState
from ..callbacks import StopController

def start_app() -> int:
    app_qt = QApplication([])
    paths = AppConfigPaths()
    app = NeuroTrainerApp(paths, ConfigStore(paths), ModelsRegistry(), PresetRegistry())
    app.run()
    win = MainWindow(app)
    win.show()
    return app_qt.exec()

class MainWindow(QMainWindow):
    def __init__(self, app: NeuroTrainerApp):
        super().__init__()
        self.setWindowTitle("model_trainer — YOLO GUI")
        self.resize(1180, 800)
        self.app = app

        self._jsonl_timer = QTimer(self); self._jsonl_timer.setInterval(1000); self._jsonl_timer.timeout.connect(self._tick_jsonl)
        self._worker: WorkerThread | None = None
        self._win_state = WindowState()

        # menu
        act_help = QAction("Help", self); act_help.triggered.connect(self._show_help)
        self.menuBar().addAction(act_help)

        self.tabs = QTabWidget(self); self.setCentralWidget(self.tabs)

        # Settings
        self.tab_settings = QWidget(); self.tabs.addTab(self.tab_settings, "Settings")
        v1 = QVBoxLayout(self.tab_settings)
        v1.addWidget(self._build_model_group())
        v1.addWidget(self._build_profile_group())
        v1.addWidget(self._build_paths_group())
        v1.addWidget(self._build_hypers_group())
        v1.addLayout(self._build_actions())

        # Progress
        self.tab_progress = QWidget(); self.tabs.addTab(self.tab_progress, "Progress")
        v2 = QVBoxLayout(self.tab_progress)
        self.progress_bar = QProgressBar(); self.progress_bar.setRange(0, 100)
        self.progress_text = QTextEdit(); self.progress_text.setReadOnly(True)
        # Stop
        self.btn_stop = QPushButton("Stop"); self.btn_stop.setEnabled(False); self.btn_stop.clicked.connect(self._on_stop)
        v2.addWidget(self.progress_bar); v2.addWidget(self.progress_text); v2.addWidget(self.btn_stop, alignment=Qt.AlignRight)

        # Charts
        self.tab_charts = QWidget(); self.tabs.addTab(self.tab_charts, "Charts")
        v3 = QVBoxLayout(self.tab_charts)
        self.fig = Figure(figsize=(6, 4)); self.canvas = FigureCanvas(self.fig)
        v3.addWidget(self.canvas)
        self._chart_task = None; self._chart_data = []

        # init
        self._load_from_state()
        self._update_valid()
        # восстановим положение окна
        g = self._win_state.restore_geometry()
        if g is not None: self.restoreGeometry(g)
        st = self._win_state.restore_state()
        if st is not None: self.restoreState(st)
        self.tabs.setCurrentIndex(self._win_state.restore_tab(0))

    # -------- save window state --------
    def closeEvent(self, e):
        self._win_state.save(self.saveGeometry(), self.saveState(), self.tabs.currentIndex())
        super().closeEvent(e)

    # -------- UI builders --------
    def _build_model_group(self) -> QGroupBox:
        box = QGroupBox("Модель"); g = QGridLayout(box)
        self.rb_file = QRadioButton("Выбрать файл .pt"); self.rb_list = QRadioButton("Выбрать из списка"); self.rb_list.setChecked(True)
        self.btn_browse = QPushButton("Обзор…"); self.btn_browse.clicked.connect(self._browse_weights)
        self.cmb_models = QComboBox(); self.cmb_models.addItems(ModelsRegistry.get_all_models())
        self.lbl_task = QLabel("Тип: —")
        self.cmb_task_override = QComboBox(); self.cmb_task_override.addItems(["(авто)", "classifier", "detector", "segmentator"]); self.cmb_task_override.setEnabled(False)
        self.rb_file.toggled.connect(self._on_model_source_changed)
        g.addWidget(self.rb_list, 0, 0); g.addWidget(self.cmb_models, 0, 1, 1, 3)
        g.addWidget(self.rb_file, 1, 0); g.addWidget(self.btn_browse, 1, 1)
        self.le_weights = QLineEdit(); self.le_weights.setPlaceholderText("путь к .pt"); self.le_weights.setReadOnly(True)
        g.addWidget(self.le_weights, 1, 2, 1, 2)
        g.addWidget(self.lbl_task, 2, 0); g.addWidget(self.cmb_task_override, 2, 1)
        return box

    def _build_profile_group(self) -> QGroupBox:
        box = QGroupBox("Тип и профиль"); h = QHBoxLayout(box)
        self.lbl_task_detected = QLabel("Определённый тип: —"); h.addWidget(self.lbl_task_detected, 1)
        self.cmb_profile = QComboBox(); self.cmb_profile.addItems(["fast", "quality", "low_vram"]); self.cmb_profile.setToolTip("fast — быстро; quality — лучше, дольше; low_vram — экономия VRAM")
        self.cmb_profile.currentIndexChanged.connect(self._update_valid)
        # device selector
        h.addWidget(QLabel("Устройство:"))
        self.cmb_device = QComboBox(); self.cmb_device.addItems(["auto", "cuda", "mps", "cpu"])
        h.addWidget(self.cmb_device)
        h.addWidget(QLabel("Профиль:")); h.addWidget(self.cmb_profile)
        return box

    def _build_paths_group(self) -> QGroupBox:
        box = QGroupBox("Пути и имя эксперимента"); g = QGridLayout(box)
        g.addWidget(QLabel("data:"), 0, 0); self.le_data = QLineEdit(); g.addWidget(self.le_data, 0, 1, 1, 3)
        g.addWidget(QLabel("project_dir:"), 1, 0); self.le_project = QLineEdit(self.app._paths.default_runs_dir.as_posix()); g.addWidget(self.le_project, 1, 1, 1, 3)
        g.addWidget(QLabel("name (опц.):"), 2, 0); self.le_name = QLineEdit(); self.le_name.setPlaceholderText("если пусто — auto runN"); g.addWidget(self.le_name, 2, 1)
        self.cb_auto_inc = QCheckBox("Авто-суффикс, если занято"); self.cb_auto_inc.setChecked(True); g.addWidget(self.cb_auto_inc, 2, 2)
        self.btn_load = QPushButton("Загрузить пресет"); self.btn_load.clicked.connect(self._load_yaml)
        self.btn_save = QPushButton("Сохранить пресет"); self.btn_save.clicked.connect(self._save_yaml)
        g.addWidget(self.btn_load, 3, 2); g.addWidget(self.btn_save, 3, 3)
        return box

    def _build_hypers_group(self) -> QGroupBox:
        box = QGroupBox("Гиперпараметры"); g = QGridLayout(box)
        self.le_epochs = QLineEdit(); self.le_epochs.setValidator(BoundedIntValidator(1, 1000))
        self.le_patience = QLineEdit(); self.le_patience.setValidator(BoundedIntValidator(0, 300))
        self.le_batch = QLineEdit(); self.le_batch.setValidator(BoundedIntValidator(1, 1024))
        self.le_imgsz = ImgSizeLineEdit()
        self.cb_rect = QCheckBox("rect (только DET/SEG)"); self.cb_multiscale = QCheckBox("multi_scale (только DET/SEG)")
        self.le_epochs.setToolTip("Сколько «кругов» обучения. 20–40 быстро; 80–150 качество.")
        self.le_patience.setToolTip("Сколько эпох подряд терпим без улучшений. 5–30 — обычно.")
        self.le_batch.setToolTip("Сколько изображений за раз. Больше — быстрее на мощном GPU, но нужна память.")
        self.le_imgsz.setToolTip("CLS: одно число (квадрат). DET/SEG: число или 'HxW'. Округлим до кратности 32.")
        g.addWidget(QLabel("epochs:"), 0, 0); g.addWidget(self.le_epochs, 0, 1)
        g.addWidget(QLabel("patience:"), 0, 2); g.addWidget(self.le_patience, 0, 3)
        g.addWidget(QLabel("batch:"), 1, 0); g.addWidget(self.le_batch, 1, 1)
        g.addWidget(QLabel("imgsz:"), 1, 2); g.addWidget(self.le_imgsz, 1, 3)
        g.addWidget(self.cb_rect, 2, 0); g.addWidget(self.cb_multiscale, 2, 1)
        for w in (self.le_epochs, self.le_patience, self.le_batch, self.le_imgsz, self.cb_rect, self.cb_multiscale):
            if hasattr(w, "textChanged"): w.textChanged.connect(self._update_valid)
            else: w.stateChanged.connect(self._update_valid)
        return box

    def _build_actions(self):
        h = QHBoxLayout()
        self.btn_validate = QPushButton("Валидация"); self.btn_validate.clicked.connect(self._on_validate)
        self.btn_start = QPushButton("Старт"); self.btn_start.clicked.connect(self._on_start)
        self.btn_open_folder = QPushButton("Открыть папку результата"); self.btn_open_folder.clicked.connect(self._open_run_dir)
        self.btn_open_log = QPushButton("Открыть лог"); self.btn_open_log.clicked.connect(self._open_log)
        self.btn_help = QPushButton("HELP"); self.btn_help.clicked.connect(self._show_help)
        h.addWidget(self.btn_validate); h.addWidget(self.btn_start); h.addStretch(1)
        h.addWidget(self.btn_open_log); h.addWidget(self.btn_open_folder); h.addWidget(self.btn_help)
        return h

    # ---------- Handlers ----------
    def _on_model_source_changed(self, checked: bool):
        by_file = self.rb_file.isChecked()
        self.btn_browse.setEnabled(by_file); self.le_weights.setEnabled(by_file); self.cmb_models.setEnabled(not by_file)
        self.cmb_task_override.setEnabled(by_file)
        self._update_valid()

    def _browse_weights(self):
        file, _ = QFileDialog.getOpenFileName(self, "Выберите .pt файл", "", "PyTorch Weights (*.pt)")
        if file: self.le_weights.setText(file)
        self._update_valid()

    def _load_yaml(self):
        file, _ = QFileDialog.getOpenFileName(self, "Загрузить пресет YAML", "", "YAML (*.yaml *.yml)")
        if not file: return
        try:
            self.app.load_preset_yaml(Path(file)); self._load_from_state()
            QMessageBox.information(self, "OK", "Пресет загружен.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка YAML", str(e))
        self._update_valid()

    def _save_yaml(self):
        file, _ = QFileDialog.getSaveFileName(self, "Сохранить пресет YAML", "", "YAML (*.yaml *.yml)")
        if not file: return
        try:
            self._push_to_state(); self.app.save_preset_yaml(Path(file))
            QMessageBox.information(self, "OK", "Пресет сохранён.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    def _on_validate(self):
        # запускаем таймер логов заранее
        self._start_jsonl_timer()
        self._start_worker(self.app.validate_only, title="Валидация")

    def _on_start(self):
        self._start_jsonl_timer()
        self._start_worker(self.app.train, title="Обучение")

    def _on_stop(self):
        StopController.request_stop()
        self.btn_stop.setEnabled(False)

    # ---------- Worker orchestration ----------
    def _start_worker(self, fn, title: str):
        try:
            self._push_to_state()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка настроек", str(e)); return

        StopController.reset()
        self.btn_stop.setEnabled(True)
        self.btn_start.setEnabled(False); self.btn_validate.setEnabled(False)

        self._worker = WorkerThread(target=fn, parent=self)
        self._worker.started_ok.connect(lambda: self.tabs.setCurrentWidget(self.tab_progress))
        self._worker.finished_ok.connect(self._on_worker_ok)
        self._worker.failed.connect(self._on_worker_fail)
        self._worker.finished.connect(self._on_worker_always)
        self._worker.start()

    def _on_worker_ok(self):
        QMessageBox.information(self, "Готово", "Операция завершена.")
    def _on_worker_fail(self, err: str):
        QMessageBox.critical(self, "Ошибка", err)
    def _on_worker_always(self):
        self.btn_stop.setEnabled(False)
        self.btn_start.setEnabled(True); self.btn_validate.setEnabled(True)
        StopController.reset()

    # ---------- sync UI <-> AppState ----------
    def _load_from_state(self):
        st = self.app._state
        if st.model_name:
            self.rb_list.setChecked(True); self.cmb_models.setCurrentText(st.model_name); self.le_weights.clear()
        elif st.model_path:
            self.rb_file.setChecked(True); self.le_weights.setText(st.model_path)
        self.cmb_profile.setCurrentText(st.profile)
        self.cmb_device.setCurrentText(st.device)
        self.le_data.setText(st.data)
        self.le_project.setText(st.project_dir)
        self.le_name.setText(st.name)
        self.cb_auto_inc.setChecked(st.auto_increment_name)
        self.le_epochs.setText(str(st.epochs)); self.le_patience.setText(str(st.patience)); self.le_batch.setText(str(st.batch))
        self.le_imgsz.setText(str(st.imgsz if isinstance(st.imgsz, int) else f"{st.imgsz[0]}x{st.imgsz[1]}"))
        self.cb_rect.setChecked(st.rect); self.cb_multiscale.setChecked(st.multi_scale)
        task = self._detect_task_ui()
        self.lbl_task.setText(f"Тип: {task or '—'}"); self.lbl_task_detected.setText(f"Определённый тип: {task or '—'}")

    def _push_to_state(self):
        if self.rb_list.isChecked():
            self.app.set_model_by_name(self.cmb_models.currentText())
        else:
            path = self.le_weights.text().strip()
            if not path: raise ValueError("Не выбран .pt файл")
            self.app.set_model_by_path(Path(path))
            task_idx = self.cmb_task_override.currentIndex()
            if task_idx > 0: self.app.set_task_override(self.cmb_task_override.currentText())
            else: self.app.set_task_override(None)

        self.app.set_profile(self.cmb_profile.currentText())
        imgsz_val = self.le_imgsz.text().strip()
        if not valid_imgsz_string(imgsz_val): raise ValueError("imgsz задан некорректно")
        self.app.set_params(
            data=self.le_data.text().strip(),
            project_dir=Path(self.le_project.text().strip()),
            name=self.le_name.text().strip(),
            auto_increment_name=self.cb_auto_inc.isChecked(),
            epochs=int(self.le_epochs.text() or "0"),
            patience=int(self.le_patience.text() or "0"),
            batch=int(self.le_batch.text() or "0"),
            imgsz=(lambda s: (tuple(map(int, s.split("x"))) if "x" in s else int(s)))(imgsz_val),
            rect=self.cb_rect.isChecked(),
            multi_scale=self.cb_multiscale.isChecked(),
            device=self.cmb_device.currentText(),
        )

    # ---------- validation ----------
    def _update_valid(self):
        valid = True
        if self.rb_list.isChecked(): valid = valid and bool(self.cmb_models.currentText())
        else: valid = valid and bool(self.le_weights.text().strip())
        valid = valid and bool(self.le_data.text().strip()) and bool(self.le_project.text().strip())
        def ok_int(le: QLineEdit) -> bool: return le.hasAcceptableInput() and le.text().strip() != ""
        valid = valid and ok_int(self.le_epochs) and ok_int(self.le_patience) and ok_int(self.le_batch)
        valid = valid and valid_imgsz_string(self.le_imgsz.text())
        task = self._detect_task_ui(); det_or_seg = task in ("detector", "segmentator")
        self.cb_rect.setEnabled(det_or_seg); self.cb_multiscale.setEnabled(det_or_seg)
        self.btn_start.setEnabled(valid and (self._worker is None or not self._worker.isRunning()))
        self.btn_validate.setEnabled(valid and (self._worker is None or not self._worker.isRunning()))

    def _detect_task_ui(self) -> str | None:
        if self.rb_list.isChecked():
            name = self.cmb_models.currentText().lower()
            if "-cls.pt" in name or name.endswith("cls.pt"): return "classifier"
            if "-seg.pt" in name or name.endswith("seg.pt"): return "segmentator"
            return "detector"
        else:
            idx = self.cmb_task_override.currentIndex()
            return self.cmb_task_override.currentText() if idx > 0 else None

    # ---------- JSONL: progress + charts ----------
    def _start_jsonl_timer(self):
        p = self.app.open_last_log()
        if p: self._jsonl_timer.start()

    def _tick_jsonl(self):
        p = self.app.open_last_log()
        if not p or not p.exists():
            return
        try:
            rows = []
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    rows.append(json.loads(line))
        except Exception:
            return
        if not rows: return

        self._chart_data = rows
        last = rows[-1]
        task = str(last.get("task", "")).replace("classify","classifier").replace("detect","detector").replace("segment","segmentator")
        self._chart_task = task

        epoch = int(last.get("epoch", 0)); epochs = int(last.get("epochs", 0))
        if epochs > 0:
            perc = max(0, min(100, int(epoch * 100 / epochs)))
            self.progress_bar.setValue(perc)
        tail_lines = rows[-20:]
        self.progress_text.setPlainText("\n".join(json.dumps(r, ensure_ascii=False) for r in tail_lines))
        self._update_charts()

    def _update_charts(self):
        if not self._chart_data: return
        self.fig.clear(); ax = self.fig.add_subplot(111)
        xs = [int(r.get("epoch", 0)) for r in self._chart_data]
        if self._chart_task == "classifier":
            y1 = [float(r.get("top1", 0.0)) for r in self._chart_data]
            y2 = [float(r.get("top5", 0.0)) for r in self._chart_data]
            ax.plot(xs, y1, label="top1"); ax.plot(xs, y2, label="top5"); ax.set_ylabel("accuracy")
        else:
            y1 = [float(r.get("map50_95", 0.0)) for r in self._chart_data]
            y2 = [float(r.get("map50", 0.0)) for r in self._chart_data]
            y3 = [float(r.get("map75", 0.0)) for r in self._chart_data]
            ax.plot(xs, y1, label="mAP50-95"); ax.plot(xs, y2, label="mAP50"); ax.plot(xs, y3, label="mAP75"); ax.set_ylabel("mAP")
        ax.set_xlabel("epoch"); ax.grid(True, alpha=0.3); ax.legend()
        self.canvas.draw()
