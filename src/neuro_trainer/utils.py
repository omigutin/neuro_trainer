from __future__ import annotations
"""
Утилиты общего назначения для приложения NeuroTrainer.

Содержимое:
- AppConfigPaths — централизованные пути к важным каталогам (configs/, runs/, presets и т.п.).
- parse_imgsz(val) — парсит int или строку 'HxW' → int | (H, W).
- ensure_dir(path) — гарантирует существование директории.
- open_in_explorer(path) — открывает файл/папку в проводнике (Win/Mac/Linux).
- yaml_str_to_dict(s) — парсит YAML-строку в dict (для GUI).

Никакой логики обучения здесь нет — только общие инструменты.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Any

import yaml


class AppConfigPaths:
    """
    Хранит пути к основным каталогам приложения:
      project_root       — корень проекта
      config_dir         — <project_root>/configs
      default_runs_dir   — <project_root>/runs
      weights_dir        — <project_root>/configs/weights        (стандартные и кастомные .pt)
      presets_dir        — <project_root>/configs/presets        (стандартные пресеты)
      presets_user_dir   — <project_root>/configs/presets_user   (кастомные пресеты)
      last_session_path  — <project_root>/configs/last_session.yaml
    """

    def __init__(
        self,
        project_root: Path | None = None,
        config_subdir: str = "configs",
        runs_subdir: str = "runs",
    ) -> None:
        if project_root is None:
            # Этот файл лежит в src/neuro_trainer/utils.py
            project_root = Path(__file__).resolve().parent.parent.parent
        self.project_root: Path = Path(project_root)

        self.config_dir: Path = self.project_root / config_subdir
        self.default_runs_dir: Path = self.project_root / runs_subdir

        # Базовые директории — создаём
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.default_runs_dir.mkdir(parents=True, exist_ok=True)

        # Подкаталоги внутри configs/
        (self.config_dir / "weights").mkdir(parents=True, exist_ok=True)
        (self.config_dir / "presets").mkdir(parents=True, exist_ok=True)
        (self.config_dir / "presets_user").mkdir(parents=True, exist_ok=True)

    # --- свойства-пути ---

    @property
    def weights_dir(self) -> Path:
        """Где хранятся веса (стандартные и кастомные) — <project>/configs/weights/"""
        return self.config_dir / "weights"

    @property
    def presets_dir(self) -> Path:
        """Где лежат стандартные YAML-пресеты — <project>/configs/presets/"""
        return self.config_dir / "presets"

    @property
    def presets_user_dir(self) -> Path:
        """Где хранить пользовательские YAML-пресеты — <project>/configs/presets_user/"""
        return self.config_dir / "presets_user"

    @property
    def last_session_path(self) -> Path:
        """Путь к last_session.yaml"""
        return self.config_dir / "last_session.yaml"

    def __repr__(self) -> str:
        return (
            f"AppConfigPaths(project_root={self.project_root}, "
            f"config_dir={self.config_dir}, default_runs_dir={self.default_runs_dir})"
        )


def parse_imgsz(val: str | int | tuple[int, int]) -> int | tuple[int, int]:
    """Приводит imgsz к int или (H, W)."""
    if isinstance(val, tuple):
        if len(val) != 2 or not all(isinstance(x, int) for x in val):
            raise ValueError("imgsz tuple должен быть длиной 2 и содержать int")
        return val
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        s = val.lower().strip()
        if "x" in s:
            a, b = s.split("x", 1)
            return int(a), int(b)
        return int(s)
    raise TypeError(f"Некорректный тип для imgsz: {type(val)}")


def ensure_dir(path: Path) -> Path:
    """Создаёт директорию (включая родителей), если её нет."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def open_in_explorer(path: Path) -> None:
    """
    Открывает файл/папку в системном проводнике. Работает под Windows/Mac/Linux.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Путь не найден: {p}")

    if sys.platform.startswith("win"):
        os.startfile(p)  # type: ignore[attr-defined]
    elif sys.platform.startswith("darwin"):
        subprocess.run(["open", str(p)], check=False)
    else:
        subprocess.run(["xdg-open", str(p)], check=False)


def yaml_str_to_dict(data: str) -> dict[str, Any]:
    """
    Парсит YAML-строку в dict. Если YAML пустой — вернёт {}.
    """
    d = yaml.safe_load(data)
    return d if isinstance(d, dict) else {}
