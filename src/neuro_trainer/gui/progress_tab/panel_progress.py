from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QProgressBar, QMessageBox
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ...utils import open_in_explorer
from ...callbacks import StopController


class PanelProgress(QGroupBox):
    """Прогресс обучения: прогресс-бар, «живой» лог, графики метрик, кнопки открыть/стоп."""
    stop_clicked = Signal()

    def __init__(self, app, parent=None):
        super().__init__("Прогресс", parent)
        self.app = app

        v = QVBoxLayout(self)

        # прогресс
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setToolTip("Доля завершённых эпох.")

        # лог
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setToolTip("Последние события обучения (хвост JSONL).")

        # график
        self.fig = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setToolTip("График метрик по эпохам.")

        # кнопки
        btn_row = QHBoxLayout()
        self.btn_log = QPushButton("Открыть лог")
        self.btn_log.setToolTip("Открыть файл JSONL-лога в системной программе.")
        self.btn_log.clicked.connect(self._open_log)

        self.btn_folder = QPushButton("Открыть папку результата")
        self.btn_folder.setToolTip("Открыть папку текущего эксперимента (runs/…/runN).")
        self.btn_folder.clicked.connect(self._open_folder)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setToolTip("Отправить запрос мягкой остановки. Тренировка завершится на границе эпохи.")
        self.btn_stop.clicked.connect(self._on_stop)

        btn_row.addWidget(self.btn_log)
        btn_row.addWidget(self.btn_folder)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_stop)

        v.addWidget(self.progress)
        v.addWidget(self.text)
        v.addWidget(self.canvas)
        v.addLayout(btn_row)

        # таймер опроса
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._tick)

        # внутренняя память
        self._last_log_path: Path | None = None
        self._last_len: int = 0  # сколько строк уже показали (простая защита от лишней работы)

        # по умолчанию кнопки лог/папка неактивны — активируем когда данные появятся
        self._update_buttons_enabled()

    # -- публичный API --
    def start_polling(self):
        self.progress.setValue(0)
        self.text.clear()
        self.fig.clear(); self.canvas.draw()
        self._last_log_path = None
        self._last_len = 0
        self._update_buttons_enabled()
        self.timer.start()

    def stop_polling(self):
        self.timer.stop()
        self._update_buttons_enabled()

    # ---- buttons ----
    def _open_folder(self):
        p = self.app.open_last_run_folder()
        if not p:
            QMessageBox.information(self, "Нет данных", "Папка результата ещё не создана.")
            return
        open_in_explorer(p)

    def _open_log(self):
        p = self.app.open_last_log()
        if not p or not p.exists():
            QMessageBox.information(self, "Нет лога", "JSONL-лог отсутствует или ещё не создан.")
            return
        open_in_explorer(p)

    def _on_stop(self):
        StopController.request_stop()
        self.btn_stop.setEnabled(False)
        self.stop_clicked.emit()

    # ---- polling ----
    def _tick(self):
        log_path = self.app.open_last_log()
        if not log_path or not log_path.exists():
            self._update_buttons_enabled()
            return

        self._last_log_path = log_path
        rows = self._read_jsonl_tail(log_path, max_lines=2000)
        if not rows:
            self._update_buttons_enabled()
            return

        last = rows[-1]
        epoch = int(last.get("epoch", 0))
        epochs = int(last.get("epochs", 0))
        if epochs > 0:
            # если epoch 1..epochs → 1/epochs..1.0
            pct = max(0, min(100, int(epoch * 100 / epochs)))
            self.progress.setValue(pct)

        # хвост текста (последние 200 строк)
        tail = rows[-200:]
        self.text.setPlainText("\n".join(json.dumps(r, ensure_ascii=False) for r in tail))
        # автоскролл вниз
        self.text.moveCursor(self.text.textCursor().End)

        # график
        self._update_chart(rows)
        self._update_buttons_enabled()

    def _read_jsonl_tail(self, path: Path, max_lines: int = 2000) -> list[dict]:
        """Читает JSONL; безопасно пропускает битые строки; возвращает последние max_lines записей."""
        rows: list[dict] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        rows.append(json.loads(s))
                    except Exception:
                        # пропускаем кривую строку
                        continue
        except Exception:
            return []
        if len(rows) > max_lines:
            rows = rows[-max_lines:]
        return rows

    def _update_chart(self, rows: list[dict]):
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        xs = [int(r.get("epoch", 0)) for r in rows]
        task = str(rows[-1].get("task", "")).lower()

        if task == "classify" or task == "classifier":
            y1 = [self._safe_float(r.get("top1", 0.0)) for r in rows]
            y2 = [self._safe_float(r.get("top5", 0.0)) for r in rows]
            ax.plot(xs, y1, label="top1")
            ax.plot(xs, y2, label="top5")
            ax.set_ylabel("accuracy")
            ax.set_ylim(0.0, 1.0)
        else:
            y1 = [self._safe_float(r.get("map50_95", 0.0)) for r in rows]
            y2 = [self._safe_float(r.get("map50", 0.0)) for r in rows]
            y3 = [self._safe_float(r.get("map75", 0.0)) for r in rows]
            ax.plot(xs, y1, label="mAP50-95")
            ax.plot(xs, y2, label="mAP50")
            ax.plot(xs, y3, label="mAP75")
            ax.set_ylabel("mAP")
            ax.set_ylim(0.0, 1.0)

        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
        self.canvas.draw()

    @staticmethod
    def _safe_float(v) -> float:
        try:
            return float(v)
        except Exception:
            return 0.0

    def _update_buttons_enabled(self):
        run_dir = self.app.open_last_run_folder()
        log_path = self.app.open_last_log()

        self.btn_folder.setEnabled(bool(run_dir))
        self.btn_log.setEnabled(bool(log_path and log_path.exists()))
