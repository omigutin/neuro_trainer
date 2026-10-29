from __future__ import annotations
from typing import Tuple

from ..tasks import TaskKind

# Ultralytics требует размер входа, кратный шагу сети
STRIDE = 32


def to_stride_multiple(v: int, stride: int = STRIDE) -> int:
    """ Округлить до ближайшего кратного stride (не меньше stride). """
    return max(stride, int(round(v / stride) * stride))


def normalize_imgsz(imgsz: int | Tuple[int, int], task: TaskKind) -> int | Tuple[int, int]:
    """
    Классификация → int-квадрат, кратный 32.
    Остальные задачи → int или (H, W), каждое кратно 32.
    """
    if task is TaskKind.CLASSIFIER:
        if isinstance(imgsz, tuple):
            imgsz = max(imgsz)
        return to_stride_multiple(int(imgsz))
    if isinstance(imgsz, tuple):
        return to_stride_multiple(int(imgsz[0])), to_stride_multiple(int(imgsz[1]))
    return to_stride_multiple(int(imgsz))
