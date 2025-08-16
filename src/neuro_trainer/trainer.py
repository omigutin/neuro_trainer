from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

from ultralytics import YOLO

from .presets import Preset
from .callbacks import StopController
from .metrics import MetricsStore

# Подпапки для результатов как у Ultralytics
_TASK_SUBDIR = {"classifier": "classify", "detector": "detect", "segmentation": "segment"}


@dataclass(slots=True)
class Trainer:
    """
    Обёртка над Ultralytics YOLO:
      • создаёт папку сохранения (project_dir/task/run_name)
      • запускает train()/val()
      • пишет JSONL-лог по эпохам (для вкладки Progress)

    Примечания:
      • Параметры аугментаций передаём через getattr(..., None) — если атрибутов нет в инстансе,
        Ultralytics их просто проигнорирует (это нормально).
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
    def _build_save_dir(self) -> Path:
        task_dir = _TASK_SUBDIR.get(self.task_hint or "detector", "detect")
        d = Path(self.project_dir).resolve() / task_dir / self.run_name
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

    def _build_callbacks(self) -> dict[str, Any]:
        """
        Пользовательские callbacks Ultralytics:
          • on_train_start — очищаем JSONL
          • on_fit_epoch_end — добавляем строку с метриками
          • on_fit_epoch_start — поддержка «мягкой» остановки
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
            except Exception:
                pass

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

                dlog.open("a", encoding="utf-8").write(__import__("json").dumps(row, ensure_ascii=False) + "\n")
            except Exception:
                # Не роняем процесс из-за несовпадения внутренних структур Ultralytics
                pass

        def on_fit_epoch_start(trainer):
            if StopController.should_stop():
                # мягкий ранний стоп
                raise RuntimeError("Early stop requested")

        return {
            "on_train_start": on_train_start,
            "on_fit_epoch_end": on_fit_epoch_end,
            "on_fit_epoch_start": on_fit_epoch_start,
        }

    # ---------------------------
    # Публичные методы
    # ---------------------------
    def validate_only(self) -> None:
        save_dir = self._build_save_dir()
        model = self._make_model()
        model.val(
            data=self.data,
            imgsz=self.imgsz,
            project=self.project_dir,
            name=self.run_name,
            device=None if self.device == "auto" else self.device,
            workers=self.workers,
        )
        try:
            MetricsStore.save_final_metrics(save_dir)
        except Exception:
            pass

    def train(self) -> None:
        save_dir = self._build_save_dir()
        model = self._make_model()

        # Подмешиваем пресетные значения
        preset_kwargs = self.preset.to_train_kwargs() if hasattr(self.preset, "to_train_kwargs") else {}

        callbacks = self._build_callbacks()

        # Запуск обучения
        model.train(
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
            project=self.project_dir,
            name=self.run_name,
            verbose=True,

            # аугментации / доп-параметры (некоторые игнорируются в зависимости от задачи — это ок)
            hsv_h=getattr(self, "hsv_h", None),
            hsv_s=getattr(self, "hsv_s", None),
            hsv_v=getattr(self, "hsv_v", None),
            degrees=getattr(self, "degrees", None),
            translate=getattr(self, "translate", None),
            scale=getattr(self, "scale", None),
            shear=getattr(self, "shear", None),
            perspective=getattr(self, "perspective", None),
            flipud=getattr(self, "flipud", None),
            fliplr=getattr(self, "fliplr", None),
            mosaic=getattr(self, "mosaic", None),
            mixup=getattr(self, "mixup", None),
            copy_paste=getattr(self, "copy_paste", None),
            auto_augment=getattr(self, "auto_augment", None),
            erasing=getattr(self, "erasing", None),
            label_smoothing=getattr(self, "label_smoothing", None),
            dropout=getattr(self, "dropout", None),

            # callbacks пользователя
            callbacks=callbacks,

            # параметры из пресета
            **preset_kwargs,
        )

        # Итоговые метрики рядом с results.csv
        try:
            MetricsStore.save_final_metrics(save_dir)
        except Exception:
            pass
