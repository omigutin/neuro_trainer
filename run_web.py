#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Запуск веб-интерфейса neuro_trainer: python run_web.py [--port 8765] [--no-browser] """

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

# Пакет лежит в src/; если проект не установлен через poetry — добавляем путь
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# Только локальный доступ: интерфейс открывает файлы и запускает процессы на этом компьютере
HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def main() -> None:
    parser = argparse.ArgumentParser(description="Neuro Trainer web UI")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="порт веб-сервера")
    parser.add_argument("--no-browser", action="store_true", help="не открывать браузер")
    args = parser.parse_args()

    try:
        import uvicorn
        from neuro_trainer.web import create_app
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] not in {"fastapi", "uvicorn", "jinja2", "starlette"}:
            raise
        print("\n[ERROR] Веб-зависимости не установлены. Из корня проекта:\n")
        print("    poetry install -E web\n")
        print(f"(Техническая причина: {e})\n")
        sys.exit(1)

    url = f"http://{HOST}:{args.port}/"
    print(f"Neuro Trainer: {url}  (Ctrl+C — остановить)")
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_app(), host=HOST, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
