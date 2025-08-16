from __future__ import annotations
from pathlib import Path

from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget, QApplication, QMessageBox
from PySide6.QtCore import Qt

from ..neuro_trainer import NeuroTrainerApp
from ..config_store import ConfigStore
from ..models_registry import ModelsRegistry
from ..presets import PresetRegistry
from ..utils import AppConfigPaths

# панели
from .settings_tab.panel_model import PanelModel
from .settings_tab.panel_dataset import PanelDataset
from .settings_tab.panel_params import PanelParams
from .settings_tab.panel_advanced import PanelAdvanced
from .settings_tab.panel_actions import PanelActions
from .progress_tab.panel_progress import PanelProgress
from .help_tab.panel_help import PanelHelp


def _apply_qss(app: QApplication) -> None:
    qss_path = Path(__file__).resolve().parent / "qss" / "app.qss"
    try:
        app.setStyle("Fusion")
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))
    except Exception:
        pass


def start_app() -> int:
    app_qt = QApplication([])
    _apply_qss(app_qt)

    paths = AppConfigPaths()
    app = NeuroTrainerApp(paths, ConfigStore(paths), ModelsRegistry(), PresetRegistry())
    app.run()

    win = MainWindow(app)
    win.show()
    return app_qt.exec()


FIELD_W = 260           # ~15 символов при 10pt
BTN_W = 170             # одинаковая ширина нижних кнопок


class MainWindow(QMainWindow):
    def __init__(self, app: NeuroTrainerApp):
        super().__init__()
        self.setWindowTitle("model_trainer — YOLO GUI")
        self.app = app
        self._worker = None  # назначается в _start_worker

        # --- вкладки ---
        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)

        # --- Settings tab ---
        tab_settings = QWidget()
        v = QVBoxLayout(tab_settings)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(14)  # больше воздуха между панелями

        self.p_model = PanelModel(self.app, field_width=FIELD_W)
        self.p_dataset = PanelDataset(self.app, field_width=FIELD_W)
        self.p_params = PanelParams(field_width=FIELD_W)
        self.p_advanced = PanelAdvanced(field_width=FIELD_W)
        self.p_actions = PanelActions(self.app, btn_width=BTN_W)

        for p in (self.p_model, self.p_dataset, self.p_params, self.p_advanced):
            p.state_changed.connect(self._update_valid)

        self.p_model.task_auto_detected.connect(self._on_task_detected)
        self.p_actions.load_clicked.connect(self._on_preset_loaded)
        self.p_actions.save_clicked.connect(self._on_preset_saved)
        self.p_actions.validate_clicked.connect(self._on_validate)
        self.p_actions.start_clicked.connect(self._on_start)

        v.addWidget(self.p_model)
        v.addWidget(self.p_dataset)
        v.addWidget(self.p_params)
        v.addWidget(self.p_advanced)
        v.addWidget(self.p_actions)

        self.tabs.addTab(tab_settings, "Settings")

        # --- Progress tab ---
        self.p_progress = PanelProgress(self.app)
        self.tabs.addTab(self.p_progress, "Progress")

        # --- Help tab ---
        self.p_help = PanelHelp()
        self.tabs.addTab(self.p_help, "Help")

        # загрузка состояния и синк UI
        self._load_state_to_ui()
        self._update_valid()

        # минимально допустимый размер окна = sizeHint компоновки
        self.setMinimumSize(self.sizeHint())

    # --- sync UI <-> app ---

    def _load_state_to_ui(self):
        s = self.app._state
        # panel_model
        self.p_model.set_task("auto" if not s.task_override else s.task_override)
        self.p_model.refresh_models()
        if s.model_name:
            self.p_model.set_model_name(s.model_name)
        self.p_model.set_data_path(s.data)

        # dataset
        self.p_dataset.set_values(device=s.device, workers=s.workers,
                                  deterministic=s.deterministic,
                                  results_dir=s.project_dir, run_name=s.name)

        # params
        imgsz_str = str(s.imgsz if isinstance(s.imgsz, int) else f"{s.imgsz[0]}x{s.imgsz[1]}")
        self.p_params.set_values(s.epochs, s.patience, s.batch, imgsz_str, s.rect, s.multi_scale)

        # advanced
        self.p_advanced.set_values(s)

    def _push_ui_to_state(self):
        # модель/тип
        task_ui = self.p_model.get_task()
        self.app.set_task_override(None if task_ui == "auto" else task_ui)
        self.app.set_model_by_name(self.p_model.get_model_name_for_state())

        # пути + имя рана
        data_path = self.p_model.get_data_path()
        rdir = self.p_dataset.get_results_dir()
        rname = self.p_dataset.get_run_name()
        self.app.set_params(data=data_path, project_dir=Path(rdir), name=rname)

        # базовые параметры
        pv = self.p_params.get_values()
        imgsz = (tuple(map(int, pv["imgsz"].split("x"))) if "x" in pv["imgsz"] else int(pv["imgsz"]))
        self.app.set_params(epochs=pv["epochs"], patience=pv["patience"], batch=pv["batch"],
                            imgsz=imgsz, rect=pv["rect"], multi_scale=pv["multi_scale"])

        # advanced + aug
        self.app.set_params(**self.p_advanced.get_values())

    # --- handlers ---

    def _on_task_detected(self, task: str):
        self.p_params.enable_rect_opts(task in ("detector", "segmentation"))

    def _on_preset_loaded(self):
        self._load_state_to_ui()
        self._update_valid()

    def _on_preset_saved(self):
        pass

    def _on_validate(self):
        try:
            self._push_ui_to_state()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка настроек", str(e)); return
        self.p_progress.start_polling()
        self._start_worker(self.app.validate_only)

    def _on_start(self):
        try:
            self._push_ui_to_state()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка настроек", str(e)); return
        self.p_progress.start_polling()
        self._start_worker(self.app.train)

    def _start_worker(self, fn):
        from .worker import WorkerThread
        if hasattr(self, "_worker") and self._worker and self._worker.isRunning():
            return
        self._worker = WorkerThread(target=fn, parent=self)
        self._worker.finished_ok.connect(self._on_worker_done)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.start()

    def _on_worker_done(self):
        self.p_progress.stop_polling()

    def _on_worker_failed(self, err: str):
        self.p_progress.stop_polling()
        QMessageBox.critical(self, "Ошибка", err)

    # --- validation gating ---

    def _update_valid(self):
        ok_params = self.p_params.is_valid()
        has_model = bool(self.p_model.get_model_name_for_state())
        has_data = bool(self.p_model.get_data_path())
        valid = ok_params and has_model and has_data
        running = hasattr(self, "_worker") and self._worker and self._worker.isRunning()
        self.p_actions.btn_start.setEnabled(valid and not running)
        self.p_actions.btn_validate.setEnabled(valid and not running)
