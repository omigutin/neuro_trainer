from __future__ import annotations
from pathlib import Path
from ..weights_cache import WeightsCache

class WeightsResolver:
    def __init__(self, cache: WeightsCache) -> None:
        self.cache = cache

    def resolve(self, model_name: str, model_path: str) -> str:
        """
        Если model_name задан → ensure() в кэше (скачает стандартный или возьмёт кастомный из configs/weights).
        Если model_path задан (устаревший сценарий) → проверим наличие и вернём абсолютный путь.
        """
        if model_name:
            return str(self.cache.ensure(model_name))

        if model_path:
            p = Path(model_path).resolve()
            if not p.exists():
                raise ValueError(f"Файл весов не найден: {p}")
            return str(p)

        raise ValueError("Не выбрана модель (нет ни model_name, ни model_path).")
