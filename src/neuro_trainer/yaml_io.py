from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any


class YamlIO:
    """
        Класс для загрузки и сохранения YAML-файлов.
        Используется для:
            - сохранения пресетов (save_preset)
            - загрузки пресетов (load_preset)
            - хранения последней сессии (last_session.yaml)
        Особенности:
        - Автоматически создаёт директории, если их нет.
        - Работает только с расширением .yaml или .yml.
        - Обрабатывает ошибки и даёт понятные исключения.
    """

    @staticmethod
    def load(path: str | Path) -> dict[str, Any]:
        """
            Загружает YAML-файл в словарь.
            Args:
                path: Путь к YAML-файлу.
            Returns:
                dict: Содержимое YAML в виде словаря.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Файл YAML не найден: {path}")
        if path.suffix.lower() not in (".yaml", ".yml"):
            raise ValueError(f"Недопустимое расширение файла: {path.suffix} (ожидалось .yaml или .yml)")

        with open(path, "r", encoding="utf-8") as f:
            try:
                return yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise ValueError(f"Ошибка парсинга YAML: {e}")

    @staticmethod
    def save(data: dict[str, Any], path: str | Path) -> None:
        """
            Сохраняет словарь в YAML-файл.
            Args:
                data: Данные для сохранения.
                path: Путь к YAML-файлу.
        """
        path = Path(path)
        if path.suffix.lower() not in (".yaml", ".yml"):
            raise ValueError(f"Недопустимое расширение файла: {path.suffix} (ожидалось .yaml или .yml)")

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    @staticmethod
    def exists(path: str | Path) -> bool:
        """
            Проверяет, существует ли файл YAML.
            Args:
                path: Путь к YAML-файлу.
            Returns:
                bool: True, если файл существует.
        """
        return Path(path).exists()
