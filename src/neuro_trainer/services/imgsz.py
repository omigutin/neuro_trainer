from __future__ import annotations
from typing import Tuple

def to_stride_multiple(v: int, stride: int = 32) -> int:
    return int(round(v / stride) * stride)

def normalize_imgsz(imgsz: int | Tuple[int, int], task: str) -> int | Tuple[int, int]:
    """
    Классификация → int-квадрат, кратный 32.
    Детекция/Сегментация → int или (H,W), каждое кратно 32.
    """
    t = (task or "").lower()
    if t == "classifier":
        if isinstance(imgsz, tuple):
            imgsz = max(imgsz)
        return to_stride_multiple(int(imgsz))
    else:
        if isinstance(imgsz, tuple):
            return (to_stride_multiple(int(imgsz[0])),
                    to_stride_multiple(int(imgsz[1])))
        return to_stride_multiple(int(imgsz))
