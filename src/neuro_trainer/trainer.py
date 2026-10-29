from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from ultralytics import YOLO

from .presets import Preset
from .callbacks import StopController
from .metrics import MetricsStore

# Подпапки для результатов как у Ultralytics
_TASK_SUBDIR = {"classifier": "classify", "detector": "detect", "segmentation": "segment"}

_log = logging.getLogger(__name__)


@dataclass(slots=True)
class Trainer:
    """
    Обёртка над Ultralytics YOLO:
      • создаёт папку сохранения (project_dir/task/run_name)
      • запускает train()/val()
      • пишет JSONL-лог по эпохам (для вкладки Progress)

    Примечания:
      • Значения из GUI имеют приоритет над пресетом; пресет дополняет недостающие параметры.
      • 'device="auto"' — даём Ultralytics выбрать устройство; иначе принудительно model.to(device).
    """
    # обязательные
    weights: str
    task_hint: str                # 'classifier' | 'detector' | 'segmentation' (может прийти пустой — но мы его уже нормализуем выше)
    data: str
    project_dir: str
    run_name: str

    # базовые гиперы
    epochs: int
    patience: int
    batch: int
    imgsz: int | Tuple[int, int]
    rect: bool
    multi_scale: bool

    # железо/воспроизводимость
    seed: int
    deterministic: bool
    workers: int
    device: str                   # 'auto'|'cuda'|'mps'|'cpu'

    # пресет (подмешиваем значения в train())
    preset: Preset

    # внутренние
    save_dir: Optional[str] = None
    jsonl_log_path: Optional[Path] = None

    # ---------------------------
    # Внутренняя подготовка
    # ---------------------------
    def _task_project_dir(self) -> Path:
        """ Каталог задачи (project_dir/<task>), в нём Ultralytics создаёт папку рана. """
        task_dir = _TASK_SUBDIR.get(self.task_hint or "detector", "detect")
        return Path(self.project_dir).resolve() / task_dir

    def _build_save_dir(self) -> Path:
        d = self._task_project_dir() / self.run_name
        d.mkdir(parents=True, exist_ok=True)
        self.save_dir = str(d)
        self.jsonl_log_path = d / "train_metrics.jsonl"
        return d

    def _make_model(self) -> YOLO:
        model = YOLO(self.weights)
        # 'auto' — пускай Ultralytics сам выберет (CUDA/MPS/CPU)
        if self.device != "auto":
            model.to(self.device)
        return model

    def _build_callbacks(self) -> dict[str, Callable[[Any], None]]:
        """
        Пользовательские callbacks Ultralytics (регистрируются через model.add_callback):
          • on_train_start — очищаем JSONL
          • on_fit_epoch_end — добавляем строку с метриками и обрабатываем «мягкую» остановку
        Best-effort: API Ultralytics иногда меняется между версиями.
        """
        dlog = self.jsonl_log_path

        def _safe_get(metrics: Any, path: str, default=None):
            cur = metrics
            for p in path.split("."):
                if cur is None:
                    return default
                cur = getattr(cur, p, None) if not isinstance(cur, dict) else cur.get(p)
            return cur if cur is not None else default

        def on_train_start(trainer):
            try:
                dlog.write_text("", encoding="utf-8")
            except OSError as e:
                _log.warning("Cannot reset progress log %s: %s", dlog, e)

        def on_fit_epoch_end(trainer):
            try:
                row = {
                    "epoch": int(getattr(trainer, "epoch", 0) + 1),
                    "epochs": int(getattr(trainer, "epochs", 0)),
                    "task": (
                        "classify"
                        if self.task_hint == "classifier"
                        else ("segment" if self.task_hint == "segmentation" else "detect")
                    ),
                }
                # Метрики могут лежать либо в trainer.metrics, либо в trainer.validator.metrics
                m = getattr(trainer, "metrics", None) or getattr(getattr(trainer, "validator", None), "metrics", None)

                if self.task_hint == "classifier":
                    row["top1"] = float(_safe_get(m, "top1", 0.0))
                    row["top5"] = float(_safe_get(m, "top5", 0.0))
                else:
                    # box mAP
                    row["map50_95"] = float(_safe_get(getattr(m, "box", None), "map", 0.0))
                    row["map50"]     = float(_safe_get(getattr(m, "box", None), "map50", 0.0))
                    row["map75"]     = float(_safe_get(getattr(m, "box", None), "map75", 0.0))

                with dlog.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            except (OSError, TypeError, ValueError) as e:
                # Сбой записи прогресса не должен останавливать обучение
                _log.warning("Cannot write epoch metrics to %s: %s", dlog, e)

            # Ultralytics проверяет trainer.stop сразу после on_fit_epoch_end:
            # обучение корректно завершится после текущей эпохи (с сохранением best/last).
            if StopController.is_set():
                _log.info("Stop requested by GUI, finishing after current epoch")
                trainer.stop = True

        return {
            "on_train_start": on_train_start,
            "on_fit_epoch_end": on_fit_epoch_end,
        }

    def _build_train_kwargs(self) -> dict[str, Any]:
        """ Параметры model.train(): пресет как база, значения из GUI поверх (без дублей ключей). """
        kwargs: dict[str, Any] = self.preset.to_train_kwargs()
        kwargs.update(
            data=self.data,
            epochs=self.epochs,
            patience=self.patience,
            batch=self.batch,
            imgsz=self.imgsz,
            rect=self.rect,
            multi_scale=self.multi_scale,
            device=None if self.device == "auto" else self.device,
            workers=self.workers,
            seed=self.seed,
            deterministic=self.deterministic,
            # Папку рана уже создали сами (project_dir/<task>/run_name) — Ultralytics пишет в неё же
            project=str(self._task_project_dir()),
            name=self.run_name,
            exist_ok=True,
            verbose=True,
        )
        return kwargs

    # ---------------------------
    # Публичные методы
    # ---------------------------
    def validate_only(self) -> None:
        save_dir = self._build_save_dir()
        model = self._make_model()
        model.val(
            data=self.data,
            imgsz=self.imgsz,
            project=str(self._task_project_dir()),
            name=self.run_name,
            exist_ok=True,
            device=None if self.device == "auto" else self.device,
            workers=self.workers,
        )
        MetricsStore.save_final_metrics(save_dir)

    def train(self) -> None:
        save_dir = self._build_save_dir()
        model = self._make_model()

        # Сбрасываем запрос остановки от предыдущего запуска
        StopController.reset()
        for event, callback in self._build_callbacks().items():
            model.add_callback(event, callback)

        model.train(**self._build_train_kwargs())

        # Итоговые метрики рядом с results.csv
        MetricsStore.save_final_metrics(save_dir)
