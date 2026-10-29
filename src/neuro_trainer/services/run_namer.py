from __future__ import annotations
from pathlib import Path

from ..tasks import TaskKind, task_info


def task_dir(project_dir: Path, task: TaskKind) -> Path:
    """ Каталог задачи: <project_dir>/<имя задачи Ultralytics>. """
    return Path(project_dir) / task_info(task).ultralytics_task


def resolve_run_name(project_dir: Path, task: TaskKind, requested: str, auto_inc: bool = True) -> str:
    """
    Если requested пусто — runN.
    Если занято и auto_inc=True — добавляем суффикс 2/3/...
    """
    name = (requested or "").strip()
    tdir = task_dir(project_dir, task)
    tdir.mkdir(parents=True, exist_ok=True)

    def exists(n: str) -> bool:
        return (tdir / n).exists()

    if not name:
        n = 1
        while exists(f"run{n}"):
            n += 1
        return f"run{n}"

    if exists(name) and auto_inc:
        k = 2
        while exists(f"{name}{k}"):
            k += 1
        return f"{name}{k}"

    return name
