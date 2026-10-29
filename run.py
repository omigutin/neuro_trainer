#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import sys

# Пакеты, которые ставятся только с extra `gui`
_GUI_MODULES = {"PySide6", "shiboken6", "matplotlib"}

if __name__ == "__main__":
    try:
        from src.neuro_trainer.gui.main_window import start_app
    except ModuleNotFoundError as e:
        # Остальные ошибки импорта — не про GUI, их не маскируем
        if (e.name or "").split(".")[0] not in _GUI_MODULES:
            raise
        print("\n[ERROR] GUI-зависимости не установлены.")
        print("Установи проект с extra `gui` (из корня проекта):\n")
        print("    poetry install -E gui\n")
        print("Подробнее — README.md, раздел «Установка».")
        print(f"(Техническая причина: {e})\n")
        sys.exit(1)

    # Запуск GUI
    sys.exit(start_app())
