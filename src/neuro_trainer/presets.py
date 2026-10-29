from __future__ import annotations
"""
Профили гиперпараметров (fast / quality / low_vram) для каждой задачи.

Источник — YAML-файлы configs/presets/<prefix>_<profile>.yaml (prefix — TaskInfo.preset_prefix).
Из файла берутся только параметры каталога; значения проверяются по каталогу.
"""

from pathlib import Path
from typing import Any

import yaml

from .params_catalog import coerce_values, has_param
from .state import Profile
from .tasks import TaskKind, task_info


class PresetRegistry:
    """ Чтение профилей гиперпараметров из каталога YAML. """

    def __init__(self, presets_dir: Path) -> None:
        # Каталог стандартных профилей (configs/presets)
        self._dir = Path(presets_dir)

    def path(self, task: TaskKind, profile: Profile) -> Path:
        """ Файл профиля для задачи. """
        return self._dir / f"{task_info(task).preset_prefix}_{profile.value}.yaml"

    def get(self, task: TaskKind, profile: Profile) -> dict[str, Any]:
        """ Значения параметров профиля (только ключи каталога, приведённые к типам). """
        p = self.path(task, profile)
        if not p.is_file():
            raise FileNotFoundError(f"Preset file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Preset {p.name} must be a YAML mapping")
        unknown = [k for k in raw if not has_param(k)]
        if unknown:
            raise ValueError(f"Preset {p.name} has unknown keys: {', '.join(unknown)}")
        values, errors = coerce_values(raw)
        if errors:
            raise ValueError(f"Preset {p.name} has invalid values: {'; '.join(errors.values())}")
        return values
