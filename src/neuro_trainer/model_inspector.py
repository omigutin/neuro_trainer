from __future__ import annotations
"""
Чтение сведений о своих весах (.pt) для дообучения: задача, классы, размер входа.
Задача берётся из самого чекпойнта, а не из имени файла.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .tasks import TaskKind, parse_task


@dataclass(frozen=True, slots=True)
class WeightsInfo:
    """ Сведения о файле весов. """
    # Путь к файлу
    path: str
    # Задача, на которую обучена модель
    task: TaskKind
    # Имена классов по индексам
    class_names: dict[int, str] = field(default_factory=dict)
    # Размер входа, с которым модель обучали (если сохранён в чекпойнте)
    imgsz: Optional[int] = None
    # Есть ли в чекпойнте состояние оптимизатора/эпоха (можно продолжить прерванное обучение)
    resumable: bool = False


def inspect_weights(path: str | Path) -> WeightsInfo:
    """ Прочитать сведения о весах. Требует ultralytics. """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Weights file not found: {p}")
    if p.suffix.lower() != ".pt":
        raise ValueError(f"Expected .pt file, got: {p.name}")

    from ultralytics import YOLO  # тяжёлый импорт — только при необходимости

    model = YOLO(str(p))
    ckpt: dict[str, Any] = model.ckpt if isinstance(getattr(model, "ckpt", None), dict) else {}
    train_args: dict[str, Any] = ckpt.get("train_args") or {}
    imgsz = train_args.get("imgsz")
    names = getattr(model, "names", None) or {}
    epoch = ckpt.get("epoch", -1)
    return WeightsInfo(
        path=str(p.resolve()),
        task=parse_task(model.task),
        class_names={int(k): str(v) for k, v in dict(names).items()},
        imgsz=int(imgsz) if isinstance(imgsz, int) else None,
        resumable=isinstance(epoch, int) and epoch >= 0 and ckpt.get("optimizer") is not None,
    )
