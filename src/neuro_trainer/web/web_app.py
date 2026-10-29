from __future__ import annotations
"""
Веб-интерфейс neuro_trainer (FastAPI + Jinja2 + обычный JS).

- create_app() — самостоятельное приложение (run_web.py);
- mount(app, prefix) — подключение к другому FastAPI-приложению (например, neuroframework).
"""

import os
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Body, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..config_store import ConfigStore
from ..help_content import HelpContent
from ..job import JobKind
from ..job_manager import JobBusyError, JobManager
from ..models_registry import ModelsRegistry
from ..neuro_trainer import NeuroTrainerApp
from ..params_catalog import ADVANCED_GROUPS, PARAMS, ParamGroup
from ..presets import PresetRegistry
from ..state import ModelSource, Profile, TrainMode
from ..tasks import TASKS, TaskKind, parse_task
from ..utils import AppConfigPaths
from .native_pickers import PickerKind, pick

_HERE = Path(__file__).resolve().parent

# Подсказки для полей, которых нет в каталоге параметров обучения
FIELD_HELP: dict[str, str] = {
    "model_source": "Стандартная — готовые веса Ultralytics, скачиваются при первом запуске. "
                    "Свой файл — ваша обученная модель (.pt) для дообучения или продолжения прерванного обучения.",
    "family": "Линейка моделей. YOLO26 — новейшая (быстрее на CPU и в экспорте), YOLO11 — проверенная.",
    "task": "Что учится делать модель: класс всего изображения, рамки объектов, маски, ключевые точки "
            "или повёрнутые рамки. Для своего файла определяется из самого файла.",
    "model_name": "Размер модели: n — самая быстрая и маленькая, x — самая точная и тяжёлая. "
                  "Для начала обычно n или s.",
    "model_path": "Файл весов .pt. Для дообучения — например best.pt прошлого рана; "
                  "для продолжения прерванного — weights/last.pt этого рана.",
    "train_mode": "Дообучение — новое обучение, начиная с весов выбранного файла (новая папка рана, текущие параметры). "
                  "Продолжить прерванное — досчитать оставшиеся эпохи с last.pt с теми же параметрами в той же папке.",
    "data": "Детекция, сегментация, pose, OBB — путь к data.yaml (формат YOLO). "
            "Классификация — папка с подпапками train/ и val/, внутри — папки по классам.",
    "project_dir": "Корневая папка результатов. Ран сохраняется в <папка>/<задача>/<название>.",
    "name": "Название папки рана. Пусто — run1, run2, ...; если папка уже есть, добавляется номер.",
    "profile": "Готовый набор гиперпараметров для задачи: fast — быстро проверить, quality — максимум качества, "
               "low_vram — для видеокарт с малой памятью. «Подставить» заменяет значения в форме.",
    "user_preset": "Свои сохранённые настройки целиком (модель, датасет, параметры) — configs/presets_user.",
}

PROFILE_LABELS = {Profile.FAST: "fast — быстро", Profile.QUALITY: "quality — качество",
                  Profile.LOW_VRAM: "low_vram — мало видеопамяти"}


@dataclass(slots=True)
class WebContext:
    """ Зависимости веб-интерфейса. """
    trainer_app: NeuroTrainerApp
    jobs: JobManager
    # Фасад не потокобезопасен, а синхронные обработчики FastAPI выполняются в пуле потоков
    lock: threading.Lock = field(default_factory=threading.Lock)


def build_context(paths: Optional[AppConfigPaths] = None) -> WebContext:
    """ Собрать зависимости по умолчанию (корень проекта, configs/, runs/). """
    paths = paths or AppConfigPaths()
    app = NeuroTrainerApp(paths, ConfigStore(paths), PresetRegistry(paths.presets_dir))
    app.load_last_session()
    return WebContext(trainer_app=app, jobs=JobManager())


def _meta() -> dict[str, Any]:
    """ Описание формы: задачи, модели, группы параметров с подсказками. """
    groups = []
    for group in ParamGroup:
        groups.append({
            "key": group.name.lower(),
            "title": group.value,
            "advanced": group in ADVANCED_GROUPS,
            "params": [{
                "key": p.key, "label": p.label, "kind": p.kind.value, "default": p.default, "help": p.help,
                "min": p.min, "max": p.max, "step": p.step, "choices": list(p.choices),
                "tasks": sorted(t.value for t in p.tasks),
            } for p in PARAMS if p.group is group],
        })
    return {
        "tasks": [{"value": t.value, "label": info.label} for t, info in TASKS.items()],
        "families": [{
            "key": f.key, "label": f.label,
            "models": {t.value: f.models(t) for t in TaskKind if f.models(t)},
        } for f in ModelsRegistry.families()],
        "profiles": [{"value": p.value, "label": PROFILE_LABELS[p]} for p in Profile],
        "groups": groups,
        "help": FIELD_HELP,
        "main_metric": {t.value: info.main_metric for t, info in TASKS.items()},
    }


def create_router(ctx: WebContext, prefix: str = "") -> APIRouter:
    """ Маршруты страницы и API. prefix — путь, по которому подключён интерфейс. """
    router = APIRouter(prefix=prefix)
    templates = Jinja2Templates(directory=str(_HERE / "templates"))
    app = ctx.trainer_app

    @router.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "index.html", {"base": prefix})

    @router.get("/api/meta")
    def meta() -> dict[str, Any]:
        return _meta()

    @router.get("/api/help")
    def help_sections() -> dict[str, str]:
        return HelpContent.get_html_sections()

    @router.get("/api/state")
    def get_state() -> dict[str, Any]:
        with ctx.lock:
            return {"state": app.state_dict(), "errors": {}}

    @router.post("/api/state")
    def post_state(values: dict[str, Any] = Body(...)) -> dict[str, Any]:
        with ctx.lock:
            errors = app.update(values)
            app.save_last_session()
            return {"state": app.state_dict(), "errors": errors}

    @router.get("/api/profile")
    def profile(task: str, profile: str) -> dict[str, Any]:
        try:
            return {"values": app.profile_values(parse_task(task), Profile(profile))}
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @router.post("/api/model/inspect")
    def inspect_model(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            with ctx.lock:
                info = app.inspect_custom_model(str(body.get("path") or ""))
                app.save_last_session()
                state = app.state_dict()
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"task": info.task.value, "classes": info.class_names, "imgsz": info.imgsz,
                "resumable": info.resumable, "state": state}

    @router.post("/api/browse")
    def browse(body: dict[str, Any] = Body(...)) -> dict[str, str]:
        try:
            kind = PickerKind(body.get("kind") or PickerKind.FOLDER.value)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        try:
            return {"path": pick(kind, body.get("initial") or None)}
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.post("/api/open")
    def open_path(body: dict[str, Any] = Body(...)) -> dict[str, bool]:
        p = Path(str(body.get("path") or ""))
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Path not found: {p}")
        _open_in_explorer(p)
        return {"ok": True}

    # ---------- пользовательские пресеты ----------

    @router.get("/api/user-presets")
    def user_presets() -> dict[str, list[str]]:
        with ctx.lock:
            return {"names": app.list_user_presets()}

    @router.post("/api/user-presets/save")
    def save_user_preset(body: dict[str, Any] = Body(...)) -> dict[str, list[str]]:
        try:
            with ctx.lock:
                app.save_user_preset(str(body.get("name") or ""))
                return {"names": app.list_user_presets()}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @router.post("/api/user-presets/load")
    def load_user_preset(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            with ctx.lock:
                app.load_user_preset(str(body.get("name") or ""))
                app.save_last_session()
                return {"state": app.state_dict(), "errors": {}}
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    # ---------- задание ----------

    def _start(kind: JobKind) -> dict[str, Any]:
        try:
            with ctx.lock:
                spec = app.build_job(kind)
            view = ctx.jobs.start(spec)
        except JobBusyError as e:
            raise HTTPException(status_code=409, detail="Обучение уже идёт — дождитесь окончания или остановите.") from e
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return view.to_dict()

    @router.post("/api/job/train")
    def job_train() -> dict[str, Any]:
        return _start(JobKind.TRAIN)

    @router.post("/api/job/validate")
    def job_validate() -> dict[str, Any]:
        return _start(JobKind.VALIDATE)

    @router.post("/api/job/stop")
    def job_stop() -> dict[str, bool]:
        try:
            ctx.jobs.request_stop()
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        return {"ok": True}

    @router.post("/api/job/kill")
    def job_kill() -> dict[str, bool]:
        try:
            ctx.jobs.kill()
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        return {"ok": True}

    @router.get("/api/job/status")
    def job_status(since: int = 0) -> dict[str, Any]:
        view = ctx.jobs.status()
        return {
            "job": view.to_dict() if view else None,
            "rows": ctx.jobs.progress(since) if view else [],
            "log": ctx.jobs.log_tail() if view else "",
        }

    return router


def mount(app: FastAPI, prefix: str = "/trainer", ctx: Optional[WebContext] = None) -> WebContext:
    """ Подключить интерфейс к существующему FastAPI-приложению по пути prefix. """
    ctx = ctx or build_context()
    app.include_router(create_router(ctx, prefix))
    app.mount(f"{prefix}/static", StaticFiles(directory=str(_HERE / "static")), name=f"nt_static{prefix}")
    return ctx


def create_app(ctx: Optional[WebContext] = None) -> FastAPI:
    """ Самостоятельное приложение: интерфейс по корню «/». """
    app = FastAPI(title="neuro_trainer")
    ctx = ctx or build_context()
    app.include_router(create_router(ctx, ""))
    app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="nt_static")
    return app


def _open_in_explorer(p: Path) -> None:
    """ Открыть файл/папку в проводнике ОС. """
    if sys.platform.startswith("win"):
        os.startfile(p)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.run(["open", str(p)], check=False)
    else:
        subprocess.run(["xdg-open", str(p)], check=False)
