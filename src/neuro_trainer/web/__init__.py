""" Веб-интерфейс neuro_trainer: create_app() — самостоятельный сервер, mount() — подключение к neuroframework. """

from .web_app import WebContext, build_context, create_app, create_router, mount

__all__ = ["WebContext", "build_context", "create_app", "create_router", "mount"]
