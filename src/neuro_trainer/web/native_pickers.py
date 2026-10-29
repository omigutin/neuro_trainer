from __future__ import annotations
"""
Системные диалоги выбора папки/файла для локального веб-интерфейса.

Браузер не отдаёт серверу полный путь к файлу, поэтому диалог открывает сам сервер
(Tkinter в отдельном процессе: не блокирует сервер и не конфликтует с его потоками).
"""

import json
import os
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

# Сколько ждать выбора пользователя, секунд
PICKER_TIMEOUT_SEC = 600
# Кодировка обмена с процессом диалога: явно с обеих сторон, иначе в Windows кириллица в пути портится
_PIPE_ENCODING = "utf-8"


class PickerKind(str, Enum):
    """ Что выбираем. """
    FOLDER = "folder"
    WEIGHTS = "weights"
    YAML = "yaml"


# Фильтры файлов для диалога (подпись, маска)
_FILETYPES: dict[PickerKind, list[tuple[str, str]]] = {
    PickerKind.WEIGHTS: [("Веса PyTorch", "*.pt"), ("Все файлы", "*.*")],
    PickerKind.YAML: [("Описание датасета", "*.yaml *.yml"), ("Все файлы", "*.*")],
}

_PICKER_CODE = """
import json, os, sys
import tkinter as tk
from tkinter import filedialog
args = json.loads(os.environ["NT_PICKER_ARGS"])
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
root.lift()
root.focus_force()
if args["kind"] == "folder":
    path = filedialog.askdirectory(title=args["title"], initialdir=args["initialdir"])
else:
    path = filedialog.askopenfilename(title=args["title"], initialdir=args["initialdir"],
                                      filetypes=[tuple(x) for x in args["filetypes"]])
root.destroy()
sys.stdout.write(path or "")
"""

_TITLES: dict[PickerKind, str] = {
    PickerKind.FOLDER: "Выберите папку",
    PickerKind.WEIGHTS: "Выберите файл весов (.pt)",
    PickerKind.YAML: "Выберите data.yaml датасета",
}


def nearest_existing_dir(initial: Optional[str]) -> Path:
    """ Ближайший существующий каталог: сам путь (или его папка для файла) либо первый существующий предок. """
    if not initial:
        return Path.home()
    p = Path(initial)
    while True:
        if p.is_dir():
            return p
        if p.parent == p:
            return Path.home()
        p = p.parent


def pick(kind: PickerKind, initial: Optional[str] = None) -> str:
    """ Показать системный диалог; вернуть выбранный путь или пустую строку (отмена). """
    args = {
        "kind": kind.value,
        "title": _TITLES[kind],
        "initialdir": str(nearest_existing_dir(initial)),
        "filetypes": _FILETYPES.get(kind, []),
    }
    env = os.environ.copy()
    env["NT_PICKER_ARGS"] = json.dumps(args, ensure_ascii=False)
    env["PYTHONIOENCODING"] = _PIPE_ENCODING
    result = subprocess.run([sys.executable, "-c", _PICKER_CODE], capture_output=True, text=True,
                            encoding=_PIPE_ENCODING, errors="replace", timeout=PICKER_TIMEOUT_SEC, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Native picker failed: {result.stderr.strip()[-500:]}")
    return result.stdout.strip()
