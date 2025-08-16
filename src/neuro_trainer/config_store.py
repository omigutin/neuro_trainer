from __future__ import annotations

"""
ConfigStore — универсальное хранилище YAML-конфигураций приложения.

Задачи:
1) last_session.yaml (последняя сессия GUI) — чтение/запись.
2) Пользовательские пресеты (configs/presets_user/*.yaml) — чтение/запись/список.
3) Универсальные методы load_yaml/save_yaml для произвольных путей (например, «Загрузить пресет»).
4) «Плоское» API — хранит/отдаёт dict, не знает доменной схемы.

Стандартные пресеты (read-only) можно хранить в configs/presets/, а пользовательские — в configs/presets_user/.
"""

from pathlib import Path
from typing import Any

import yaml

from .utils import AppConfigPaths


class ConfigStore:
    """Работа с YAML-файлами конфигураций приложения."""

    LAST_SESSION_BASENAME = "last_session.yaml"

    def __init__(self, paths: AppConfigPaths) -> None:
        self.paths = paths
        self.config_dir: Path = paths.config_dir
        self.presets_user_dir: Path = paths.presets_user_dir
        self.last_session_path: Path = paths.last_session_path

    # ---------------------------
    # last_session.yaml
    # ---------------------------

    def load_last_session(self) -> dict[str, Any]:
        """Загружает last_session.yaml, если он существует (иначе {})."""
        return self._load_yaml(self.last_session_path)

    def save_last_session(self, data: dict[str, Any]) -> None:
        """Сохраняет last_session.yaml."""
        self._save_yaml(self.last_session_path, data)

    # ---------------------------
    # Пользовательские пресеты
    # ---------------------------

    def list_presets(self) -> list[str]:
        """Список имён пользовательских пресетов (без .yaml) из configs/presets_user/."""
        return sorted([p.stem for p in self.presets_user_dir.glob("*.yaml")])

    def preset_exists(self, name: str) -> bool:
        """Проверяет наличие пресета в presets_user по имени (без .yaml)."""
        return (self.presets_user_dir / f"{name}.yaml").exists()

    def load_preset(self, name: str) -> dict[str, Any]:
        """Читает пресет из presets_user по имени (без .yaml)."""
        return self._load_yaml(self.presets_user_dir / f"{name}.yaml")

    def save_preset(self, name: str, data: dict[str, Any]) -> None:
        """Сохраняет пресет в presets_user под именем (без .yaml)."""
        self._save_yaml(self.presets_user_dir / f"{name}.yaml", data)

    # ---------------------------
    # Универсальные методы для произвольного пути
    # ---------------------------

    def load_yaml(self, path: str | Path) -> dict[str, Any]:
        """Загружает YAML по переданному пути (удобно для «Загрузить пресет»)."""
        return self._load_yaml(Path(path))

    def save_yaml(self, path: str | Path, data: dict[str, Any]) -> None:
        """Сохраняет YAML по переданному пути (удобно для «Сохранить пресет»)."""
        self._save_yaml(Path(path), data)

    # ---------------------------
    # Внутренние методы
    # ---------------------------

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        """Читает YAML-файл и возвращает dict. Если нет — {}."""
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _save_yaml(path: Path, data: dict[str, Any]) -> None:
        """Сохраняет dict в YAML-файл (создаёт директории при необходимости)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
