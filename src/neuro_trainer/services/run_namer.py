from __future__ import annotations
from pathlib import Path

TASK_DIR = {"classifier": "classify", "detector": "detect", "segmentation": "segment"}

def resolve_run_name(project_dir: Path, task: str, requested: str, auto_inc: bool = True) -> str:
    """
    Если requested пусто — runN.
    Если занято и auto_inc=True — добавляем суффикс 2/3/...
    """
    name = (requested or "").strip()
    task_dir = project_dir / TASK_DIR.get((task or "").lower(), "detect")
    task_dir.mkdir(parents=True, exist_ok=True)

    def exists(n: str) -> bool:
        return (task_dir / n).exists()

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
