from __future__ import annotations
"""
WeightsCache — локальный кэш предобученных весов YOLO.

Хранение: <project>/configs/weights/<model_name>.pt
Использование:
    cache = WeightsCache(paths.weights_dir)
    local_path = cache.ensure("yolo11n.pt")  # скачает через Ultralytics и положит в кэш
    cache.is_cached("yolo11n.pt") -> bool
    cache.list_cached() -> ["yolo11n.pt", "my_custom.pt", ...]
"""

from pathlib import Path
from typing import Iterable, Optional, List, Tuple
import shutil
import glob

from ultralytics import YOLO


class WeightsCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # --------- public ---------
    def local_path(self, model_name: str) -> Path:
        return self.cache_dir / model_name

    def is_cached(self, model_name: str) -> bool:
        return self.local_path(model_name).exists()

    def ensure(self, model_name: str) -> Path:
        """
        Гарантирует наличие веса локально: если нет — скачивает через Ultralytics и
        копирует в наш кэш (имя файла — как model_name).
        Если это уже локальный .pt (лежит в cache_dir), просто возвращает его путь.
        """
        dst = self.local_path(model_name)
        if dst.exists():
            return dst

        # Триггерим загрузку у Ultralytics
        model = YOLO(model_name)

        src = _detect_source_weights_path(model, model_name)
        if src is None or not src.exists():
            raise FileNotFoundError(f"Не удалось найти путь к загруженным весам для '{model_name}'")

        shutil.copy2(src, dst)
        return dst

    def prefetch(self, models: Iterable[str]) -> List[Tuple[str, bool, str]]:
        res: List[Tuple[str, bool, str]] = []
        for m in models:
            try:
                self.ensure(m)
                res.append((m, True, "ok"))
            except Exception as e:
                res.append((m, False, str(e)))
        return res

    def list_cached(self) -> list[str]:
        return sorted([p.name for p in self.cache_dir.glob("*.pt")])

    def open_dir(self) -> Path:
        return self.cache_dir


# ---------- helpers ----------

def _detect_source_weights_path(model: YOLO, model_name: str) -> Optional[Path]:
    """
    Пытаемся понять, где Ultralytics сохранил вес. В разных версиях атрибуты отличаются,
    поэтому делаем best-effort.
    """
    # 1) Пробуем прямые атрибуты
    for attr in ("ckpt_path", "pt_path", "weights", "ckpt"):
        p = getattr(model, attr, None)
        if isinstance(p, (str, Path)):
            pp = Path(p)
            if pp.exists() and pp.suffix == ".pt":
                return pp

    # 2) Частые кэши Ultralytics + пробуем найти файл по имени
    candidates = [
        Path.home() / "AppData" / "Roaming" / "Ultralytics",  # Windows
        Path.home() / ".cache" / "Ultralytics",               # Linux
        Path.home() / ".ultralytics",                         # Linux/mac
    ]
    for base in candidates:
        cand = base / model_name
        if cand.exists():
            return cand
        # иногда файл кладут глубже; поищем по маске
        for hit in glob.glob(str(base / "**" / model_name), recursive=True):
            ph = Path(hit)
            if ph.exists():
                return ph

    return None
