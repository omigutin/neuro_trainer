from __future__ import annotations

"""
    ConfigStore — универсальное хранилище YAML-конфигураций приложения.
    Задачи:
    1. Загрузить настройки из YAML по имени (из папки configs).
    2. Сохранить настройки в YAML.
    3. Работать с "last_session.yaml" (последняя конфигурация, использованная в GUI).
    4. Проверять, есть ли такой пресет.
    5. Возвращать дефолтные настройки, если файл не найден.
    Архитектурно:
    - Этот класс ничего не знает про структуру настроек — просто хранит и отдаёт dict.
    - Пути получает через AppConfigPaths (utils.py).
    - Все YAML-файлы лежат в `paths.config_dir`.
"""

import yaml
from pathlib import Path
from typing import Any

from .utils import AppConfigPaths


class ConfigStore:
    """ Класс для работы с YAML-файлами конфигураций приложения ."""

    LAST_SESSION_FILE = "last_session.yaml"

    def __init__(self, paths: AppConfigPaths) -> None:
        self.paths = paths
        self.config_dir: Path = paths.config_dir

    # ---------------------------
    # Методы для last_session.yaml
    # ---------------------------

    def load_last_session(self) -> dict[str, Any]:
        """Загружает last_session.yaml, если он существует."""
        return self._load_yaml(self.config_dir / self.LAST_SESSION_FILE)

    def save_last_session(self, data: dict[str, Any]) -> None:
        """Сохраняет last_session.yaml."""
        self._save_yaml(self.config_dir / self.LAST_SESSION_FILE, data)

    # ---------------------------
    # Методы для пользовательских пресетов
    # ---------------------------

    def load_preset(self, name: str) -> dict[str, Any]:
        """Загружает YAML-пресет по имени (без расширения)."""
        return self._load_yaml(self.config_dir / f"{name}.yaml")

    def save_preset(self, name: str, data: dict[str, Any]) -> None:
        """Сохраняет YAML-пресет по имени (без расширения)."""
        self._save_yaml(self.config_dir / f"{name}.yaml", data)

    def list_presets(self) -> list[str]:
        """Возвращает список имён всех YAML-пресетов (без .yaml)."""
        return [
            f.stem
            for f in self.config_dir.glob("*.yaml")
            if f.name != self.LAST_SESSION_FILE
        ]

    def preset_exists(self, name: str) -> bool:
        """Проверяет, существует ли пресет с таким именем."""
        return (self.config_dir / f"{name}.yaml").exists()

    # ---------------------------
    # Внутренние методы
    # ---------------------------

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        """Читает YAML-файл и возвращает dict. Если нет — пустой dict."""
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}

    def _save_yaml(self, path: Path, data: dict[str, Any]) -> None:
        """Сохраняет dict в YAML-файл."""
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
