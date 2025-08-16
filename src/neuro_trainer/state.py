from __future__ import annotations
"""
AppState — единое описание состояния настроек тренировки (то, что редактирует GUI).
Сюда же — конвертеры state_to_dict / state_from_dict для YAML/GUI.
"""

from dataclasses import dataclass
from typing import Literal, Any

TaskType = Literal["classifier", "detector", "segmentation"]
ProfileType = Literal["fast", "quality", "low_vram"]


@dataclass(slots=True)
class AppState:
    # модель: через имя (в кэше configs/weights) или абсолютный путь (наследие)
    model_name: str = ""
    model_path: str = ""           # обычно пусто; кастом кладём в configs/weights и выбираем по имени
    task_override: str = ""        # '' = auto (определим из модели)
    profile: ProfileType = "fast"

    # пути/имена
    data: str = ""
    project_dir: str = ""
    name: str = ""
    auto_increment_name: bool = True  # всегда включено в UX

    # базовые гиперпараметры
    epochs: int = 50
    patience: int = 50
    batch: int = 16
    imgsz: int | tuple[int, int] = 384

    # флаги (детекция/сегментация)
    rect: bool = False
    multi_scale: bool = False

    # железо/воспроизводимость
    seed: int = 0
    deterministic: bool = True
    workers: int = 4
    device: str = "auto"           # auto|cuda|mps|cpu

    # --- аугментации (в основном det/seg; часть применима и к cls) ---
    # цвет
    hsv_h: float = 0.015
    hsv_s: float = 0.70
    hsv_v: float = 0.40
    # геометрия
    degrees: float = 0.0
    translate: float = 0.10
    scale: float = 0.50
    shear: float = 0.0
    perspective: float = 0.0
    # флипы
    flipud: float = 0.0
    fliplr: float = 0.5
    # миксы
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0   # seg-only
    # CLS-специфика
    auto_augment: str = "randaugment"
    erasing: float = 0.4
    label_smoothing: float = 0.0
    dropout: float = 0.0


def state_to_dict(s: AppState) -> dict[str, Any]:
    imgsz = s.imgsz
    return {
        "model_name": s.model_name,
        "pretrained_model_path": s.model_path,
        "task_override": s.task_override,
        "profile": s.profile,
        "data": s.data,
        "project_dir": s.project_dir,
        "name": s.name,
        "auto_increment_name": s.auto_increment_name,
        "epochs": s.epochs,
        "patience": s.patience,
        "batch": s.batch,
        "imgsz": imgsz if isinstance(imgsz, int) else f"{imgsz[0]}x{imgsz[1]}",
        "rect": s.rect,
        "multi_scale": s.multi_scale,
        "seed": s.seed,
        "deterministic": s.deterministic,
        "workers": s.workers,
        "device": s.device,
        "hsv_h": s.hsv_h, "hsv_s": s.hsv_s, "hsv_v": s.hsv_v,
        "degrees": s.degrees, "translate": s.translate, "scale": s.scale, "shear": s.shear, "perspective": s.perspective,
        "flipud": s.flipud, "fliplr": s.fliplr,
        "mosaic": s.mosaic, "mixup": s.mixup, "copy_paste": s.copy_paste,
        "auto_augment": s.auto_augment, "erasing": s.erasing, "label_smoothing": s.label_smoothing, "dropout": s.dropout,
    }


def state_from_dict(d: dict[str, Any], default_runs_dir: str) -> AppState:
    imgsz = d.get("imgsz", 384)
    if isinstance(imgsz, str) and "x" in imgsz.lower():
        a, b = imgsz.lower().split("x", 1)
        imgsz_parsed: int | tuple[int, int] = (int(a), int(b))
    else:
        imgsz_parsed = int(imgsz)

    return AppState(
        model_name=str(d.get("model_name", "")),
        model_path=str(d.get("pretrained_model_path", "")),
        task_override=str(d.get("task_override", "")),
        profile=str(d.get("profile", "fast")),
        data=str(d.get("data", "")),
        project_dir=str(d.get("project_dir", default_runs_dir)),
        name=str(d.get("name", "")),
        auto_increment_name=bool(d.get("auto_increment_name", True)),
        epochs=int(d.get("epochs", 50)),
        patience=int(d.get("patience", 50)),
        batch=int(d.get("batch", 16)),
        imgsz=imgsz_parsed,
        rect=bool(d.get("rect", False)),
        multi_scale=bool(d.get("multi_scale", False)),
        seed=int(d.get("seed", 0)),
        deterministic=bool(d.get("deterministic", True)),
        workers=int(d.get("workers", 4)),
        device=str(d.get("device", "auto")),
        hsv_h=float(d.get("hsv_h", 0.015)), hsv_s=float(d.get("hsv_s", 0.70)), hsv_v=float(d.get("hsv_v", 0.40)),
        degrees=float(d.get("degrees", 0.0)), translate=float(d.get("translate", 0.10)), scale=float(d.get("scale", 0.50)),
        shear=float(d.get("shear", 0.0)), perspective=float(d.get("perspective", 0.0)),
        flipud=float(d.get("flipud", 0.0)), fliplr=float(d.get("fliplr", 0.5)),
        mosaic=float(d.get("mosaic", 1.0)), mixup=float(d.get("mixup", 0.0)), copy_paste=float(d.get("copy_paste", 0.0)),
        auto_augment=str(d.get("auto_augment", "randaugment")), erasing=float(d.get("erasing", 0.4)),
        label_smoothing=float(d.get("label_smoothing", 0.0)), dropout=float(d.get("dropout", 0.0)),
    )
