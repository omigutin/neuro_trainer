from __future__ import annotations
from typing import Callable
from PySide6.QtCore import QThread, Signal
import traceback


class WorkerThread(QThread):
    """
    Запускает длительную операцию (train/validate) в отдельном потоке.

    Сигналы:
      - started_ok(): перед самым запуском целевой функции
      - finished_ok(): если функция завершилась без исключений
      - failed(str): если словили исключение; текст содержит краткую ошибку и traceback

    Использование:
        w = WorkerThread(target=app.train)
        w.started_ok.connect(...)
        w.finished_ok.connect(...)
        w.failed.connect(...)
        w.start()
    """
    started_ok = Signal()
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(self, target: Callable[[], None], parent=None) -> None:
        super().__init__(parent)
        self._target = target

    def run(self) -> None:
        try:
            self.started_ok.emit()
            self._target()
            self.finished_ok.emit()
        except Exception as e:
            tb = traceback.format_exc()
            # Соединяем краткое сообщение и трейсбек, чтобы диалог показывал максимум пользы
            self.failed.emit(f"{e}\n\n{tb}")
