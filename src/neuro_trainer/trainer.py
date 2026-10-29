from __future__ import annotations
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .callbacks import StopFlag
from .metrics import MetricsStore
from .tasks import TaskKind, task_info

_log = logging.getLogger(__name__)

# Имена служебных файлов в папке рана
PROGRESS_FILE = "train_metrics.jsonl"
STOP_FILE = "stop.request"
VAL_METRICS_FILE = "val_metrics.json"


@dataclass(slots=True)
class Trainer:
    """
    Обёртка над Ultralytics YOLO:
      • папка рана: <project_dir>/<задача Ultralytics>/<run_name> (при resume — папка исходного рана);
      • train() / val();
      • JSONL-лог метрик по эпохам (для графика прогресса);
      • «мягкая» остановка по файлу-флагу stop.request после текущей эпохи.
    """
    # Путь к весам: стандартные (скачанные) или свои; при resume — last.pt прерванного рана
    weights: str
    # Задача обучения
    task: TaskKind
    # Датасет: data.yaml или папка (классификация)
    data: str
    # Корень результатов
    project_dir: str
    # Имя рана
    run_name: str
    # Аргументы model.train()/val() из каталога параметров (без data/project/name)
    train_args: dict[str, Any] = field(default_factory=dict)
    # Продолжить прерванное обучение (Ultralytics берёт параметры из чекпойнта)
    resume: bool = False

    @property
    def save_dir(self) -> Path:
        """ Папка рана. """
        if self.resume:
            # .../<run>/weights/last.pt → .../<run>
            return Path(self.weights).resolve().parent.parent
        return self._task_project_dir() / self.run_name

    @property
    def progress_path(self) -> Path:
        """ JSONL-лог метрик по эпохам. """
        return self.save_dir / PROGRESS_FILE

    @property
    def stop_flag(self) -> StopFlag:
        """ Флаг «мягкой» остановки. """
        return StopFlag(self.save_dir / STOP_FILE)

    # ---------------------------
    # Внутренняя подготовка
    # ---------------------------
    def _task_project_dir(self) -> Path:
        """ Каталог задачи (project_dir/<task>), в нём Ultralytics создаёт папку рана. """
        return Path(self.project_dir).resolve() / task_info(self.task).ultralytics_task

    def _make_model(self) -> Any:
        from ultralytics import YOLO  # тяжёлый импорт — только в процессе обучения
        return YOLO(self.weights)

    def _build_callbacks(self) -> dict[str, Callable[[Any], None]]:
        """
        Пользовательские callbacks Ultralytics (регистрируются через model.add_callback):
          • on_train_start — очищаем JSONL (кроме resume);
          • on_fit_epoch_end — строка с метриками эпохи и обработка «мягкой» остановки.
        """
        progress = self.progress_path
        stop_flag = self.stop_flag
        resume = self.resume

        def on_train_start(trainer: Any) -> None:
            if resume:
                return
            try:
                progress.write_text("", encoding="utf-8")
            except OSError as e:
                _log.warning("Cannot reset progress log %s: %s", progress, e)

        def on_fit_epoch_end(trainer: Any) -> None:
            try:
                with progress.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(epoch_row(trainer), ensure_ascii=False) + "\n")
            except (OSError, TypeError, ValueError) as e:
                # Сбой записи прогресса не должен останавливать обучение
                _log.warning("Cannot write epoch metrics to %s: %s", progress, e)

            # Ultralytics проверяет trainer.stop сразу после on_fit_epoch_end:
            # обучение корректно завершится после текущей эпохи (best/last уже сохранены).
            if stop_flag.is_set():
                _log.info("Stop requested, finishing after current epoch")
                trainer.stop = True

        return {"on_train_start": on_train_start, "on_fit_epoch_end": on_fit_epoch_end}

    def _build_train_kwargs(self) -> dict[str, Any]:
        """ Аргументы model.train(). """
        if self.resume:
            return {"resume": True}
        kwargs: dict[str, Any] = dict(self.train_args)
        kwargs.update(
            data=self.data,
            # Папку рана создаём сами (project_dir/<task>/run_name) — Ultralytics пишет в неё же
            project=str(self._task_project_dir()),
            name=self.run_name,
            exist_ok=True,
            verbose=True,
        )
        return kwargs

    # ---------------------------
    # Публичные методы
    # ---------------------------
    def validate_only(self) -> dict[str, float]:
        """ Проверка модели на датасете без обучения; метрики — в val_metrics.json папки рана. """
        save_dir = self.save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        model = self._make_model()
        allowed = ("imgsz", "batch", "device", "workers", "single_cls")
        val_args = {k: v for k, v in self.train_args.items() if k in allowed}
        metrics = model.val(data=self.data, project=str(self._task_project_dir()), name=self.run_name,
                            exist_ok=True, **val_args)
        results = {k: float(v) for k, v in dict(getattr(metrics, "results_dict", {}) or {}).items()}
        (save_dir / VAL_METRICS_FILE).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return results

    def train(self) -> None:
        """ Обучение (или продолжение прерванного при resume=True). """
        save_dir = self.save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        # Сбрасываем запрос остановки от предыдущего запуска
        self.stop_flag.reset()

        model = self._make_model()
        for event, callback in self._build_callbacks().items():
            model.add_callback(event, callback)

        model.train(**self._build_train_kwargs())

        # Итоговые метрики рядом с results.csv
        MetricsStore.save_final_metrics(save_dir)


def epoch_row(trainer: Any) -> dict[str, Any]:
    """
    Строка JSONL по итогам эпохи: номер эпохи, потери train, метрики val, скорость обучения.
    Ключи — как в results.csv Ultralytics (metrics/mAP50-95(B), train/box_loss, lr/pg0, ...).
    """
    row: dict[str, Any] = {
        "epoch": int(getattr(trainer, "epoch", 0)) + 1,
        "epochs": int(getattr(trainer, "epochs", 0)),
        "time": round(time.time(), 1),
    }
    values: dict[str, Any] = {}
    label_loss_items = getattr(trainer, "label_loss_items", None)
    tloss = getattr(trainer, "tloss", None)
    if callable(label_loss_items) and tloss is not None:
        values.update(label_loss_items(tloss, prefix="train"))
    values.update(getattr(trainer, "metrics", None) or {})
    values.update(getattr(trainer, "lr", None) or {})
    for key, value in values.items():
        number = _to_float(value)
        if number is not None:
            row[str(key)] = number
    return row


def _to_float(value: Any) -> float | None:
    """ Число (в т.ч. тензор из одного элемента) → float; нечисловое или NaN → None. """
    try:
        number = float(value.item() if hasattr(value, "item") else value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) or math.isinf(number) else round(number, 6)
