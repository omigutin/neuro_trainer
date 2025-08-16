from __future__ import annotations
"""
YamlIO — загрузка/сохранение YAML-файлов:
- сохранение пресетов
- загрузка пресетов
- хранение last_session.yaml

Особенности:
- Автоматически создаёт директорию назначения при сохранении.
- Проверяет расширение (.yaml или .yml).
- Даёт понятные ошибки при парсинге.
"""

from pathlib import Path
from typing import Any
import yaml


class YamlIO:
    @staticmethod
    def load(path: str | Path) -> dict[str, Any]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Файл YAML не найден: {path}")
        if path.suffix.lower() not in (".yaml", ".yml"):
            raise ValueError(f"Недопустимое расширение файла: {path.suffix} (ожидалось .yaml или .yml)")
        with path.open("r", encoding="utf-8") as f:
            try:
                return yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise ValueError(f"Ошибка парсинга YAML: {e}")

    @staticmethod
    def save(data: dict[str, Any], path: str | Path) -> None:
        path = Path(path)
        if path.suffix.lower() not in (".yaml", ".yml"):
            raise ValueError(f"Недопустимое расширение файла: {path.suffix} (ожидалось .yaml или .yml)")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    @staticmethod
    def exists(path: str | Path) -> bool:
        return Path(path).exists()
