from __future__ import annotations

"""
    Утилиты общего назначения для приложения NeuroTrainer.
    - AppConfigPaths — хранит и вычисляет пути к важным каталогам (config_dir, runs_dir и т.д.).
    - parse_imgsz — переводит строку вида "640x480" в tuple[int, int] или int.
    - ensure_dir — создаёт директорию (с родителями).
    - open_in_explorer — открывает папку/файл в системном проводнике.
    - yaml_str_to_dict — парсит YAML-строку в dict (например, для GUI).
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
        - project_root       — корень исходников/репозитория
        - config_dir         — где лежат last_session.yaml и пользовательские YAML-пресеты
        - default_runs_dir   — корень для результатов (runs/)
    """

    def __init__(
        self,
        project_root: Path | None = None,
        config_subdir: str = "configs",
        runs_subdir: str = "runs",
    ) -> None:
        if project_root is None:
            # считаем, что этот файл в src/neuro_trainer/utils.py
            project_root = Path(__file__).resolve().parent.parent.parent
        self.project_root: Path = project_root

        self.config_dir: Path = self.project_root / config_subdir
        self.default_runs_dir: Path = self.project_root / runs_subdir

        # создаём каталоги, если нет
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.default_runs_dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return (
            f"AppConfigPaths(project_root={self.project_root}, "
            f"config_dir={self.config_dir}, default_runs_dir={self.default_runs_dir})"
        )


def parse_imgsz(val: str | int | tuple[int, int]) -> int | tuple[int, int]:
    """ Приводит imgsz к int или (H, W). """
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
    """ Создаёт директорию (включая родителей), если её нет. """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def open_in_explorer(path: Path) -> None:
    """
        Открывает файл/папку в системном проводнике.
        Работает под Windows/Mac/Linux.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Путь не найден: {p}")

    if sys.platform.startswith("win"):
        os.startfile(p)  # type: ignore
    elif sys.platform.startswith("darwin"):
        subprocess.run(["open", str(p)])
    else:
        subprocess.run(["xdg-open", str(p)])


def yaml_str_to_dict(data: str) -> dict[str, Any]:
    """
        Парсит YAML-строку в dict.
        Если YAML пустой — вернёт пустой dict.
    """
    d = yaml.safe_load(data)
    return d if isinstance(d, dict) else {}
