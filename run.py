#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import sys

if __name__ == "__main__":
    try:
        from src.neuro_trainer.gui.main_window import start_app
    except ImportError as e:
        print("\n[ERROR] GUI-зависимости не установлены.")
        print("Чтобы запустить графический интерфейс, установи проект с extra `gui`:\n")
        print("    poetry install -E gui\n")
        print("Или добавь PySide6 напрямую:")
        print("    poetry add PySide6@^6.9.1\n")
        print(f"(Техническая причина: {e})\n")
        sys.exit(1)

    # Запуск GUI
    sys.exit(start_app())
