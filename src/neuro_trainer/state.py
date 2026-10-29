from __future__ import annotations
"""
AppState — состояние настроек обучения (то, что редактирует интерфейс).

Параметры обучения хранятся словарём params по ключам каталога (params_catalog.PARAMS).
В YAML (last_session.yaml, пользовательские пресеты) всё лежит «плоско» — тот же формат,
что и раньше, поэтому старые файлы читаются.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .params_catalog import coerce_values, default_values, format_imgsz, has_param
from .tasks import TaskKind, parse_task

_log = logging.getLogger(__name__)


# Стандартные веса по умолчанию для нового состояния
DEFAULT_MODEL_NAME = "yolo26n.pt"


class ModelSource(str, Enum):
    """ Откуда берётся модель для обучения. """
    # Стандартные веса Ultralytics по имени (скачиваются при первом запуске)
    STANDARD = "standard"
    # Свой файл .pt (дообучение своей модели)
    CUSTOM = "custom"


class TrainMode(str, Enum):
    """ Режим обучения от своих весов. """
    # Новое обучение, начальные веса — выбранный файл
    FINETUNE = "finetune"
    # Продолжить прерванное обучение с last.pt (те же параметры, та же папка)
    RESUME = "resume"


class Profile(str, Enum):
    """ Профиль готовых гиперпараметров (configs/presets/<task>_<profile>.yaml). """
    FAST = "fast"
    QUALITY = "quality"
    LOW_VRAM = "low_vram"


@dataclass(slots=True)
class AppState:
    """ Настройки обучения. """
    # Источник модели: стандартные веса или свой файл
    model_source: ModelSource = ModelSource.STANDARD
    # Имя стандартных весов (yolo26n.pt и т.п.)
    model_name: str = DEFAULT_MODEL_NAME
    # Путь к своему файлу весов
    model_path: str = ""
    # Режим для своих весов: дообучение или продолжение прерванного
    train_mode: TrainMode = TrainMode.FINETUNE
    # Задача (для своих весов определяется из файла)
    task: TaskKind = TaskKind.DETECTOR
    # Профиль, из которого последний раз подставлялись гиперпараметры
    profile: Profile = Profile.FAST

    # Датасет: data.yaml (детекция/сегментация/pose/obb) или папка (классификация)
    data: str = ""
    # Корень результатов; ран сохраняется в <project_dir>/<task>/<name>
    project_dir: str = ""
    # Имя рана (пусто — run1, run2, ...)
    name: str = ""
    # Добавлять суффикс 2/3/..., если папка рана уже существует
    auto_increment_name: bool = True

    # Параметры обучения по ключам каталога
    params: dict[str, Any] = field(default_factory=default_values)


# Ключи верхнего уровня (не параметры каталога)
_STATE_KEYS = ("model_source", "model_name", "model_path", "train_mode", "task", "profile",
               "data", "project_dir", "name", "auto_increment_name")


def state_to_dict(s: AppState) -> dict[str, Any]:
    """ Состояние → плоский dict для YAML/веба. """
    out: dict[str, Any] = {
        "model_source": s.model_source.value,
        "model_name": s.model_name,
        "model_path": s.model_path,
        "train_mode": s.train_mode.value,
        "task": s.task.value,
        "profile": s.profile.value,
        "data": s.data,
        "project_dir": s.project_dir,
        "name": s.name,
        "auto_increment_name": s.auto_increment_name,
    }
    for key, value in s.params.items():
        out[key] = format_imgsz(value) if key == "imgsz" else value
    return out


def state_from_dict(d: dict[str, Any], default_runs_dir: str) -> AppState:
    """
    Плоский dict → состояние. Понимает старый формат
    (pretrained_model_path, task_override, multi_scale: bool, label_smoothing).
    Ошибочные значения параметров заменяются значениями по умолчанию с предупреждением в лог.
    """
    model_path = str(d.get("model_path") or d.get("pretrained_model_path") or "")
    source_raw = d.get("model_source") or (ModelSource.CUSTOM.value if model_path else ModelSource.STANDARD.value)

    task_raw = d.get("task") or d.get("task_override") or ""
    task = parse_task(task_raw) if task_raw else _task_from_model_name(str(d.get("model_name") or ""))

    raw_params = {k: v for k, v in d.items() if has_param(k)}
    # В Ultralytics 8.3 multi_scale был флагом; True соответствовал ±50%
    if isinstance(raw_params.get("multi_scale"), bool):
        raw_params["multi_scale"] = 0.5 if raw_params["multi_scale"] else 0.0
    params, errors = coerce_values(raw_params)
    for key, message in errors.items():
        _log.warning("Ignored invalid saved value: %s", message)

    ignored = [k for k in d if k not in _STATE_KEYS and not has_param(k)
               and k not in ("pretrained_model_path", "task_override")]
    if ignored:
        _log.warning("Ignored unknown saved keys: %s", ", ".join(sorted(ignored)))

    return AppState(
        model_source=ModelSource(source_raw),
        model_name=str(d.get("model_name") or DEFAULT_MODEL_NAME),
        model_path=model_path,
        train_mode=TrainMode(d.get("train_mode") or TrainMode.FINETUNE.value),
        task=task,
        profile=Profile(d.get("profile") or Profile.FAST.value),
        data=str(d.get("data") or ""),
        project_dir=str(d.get("project_dir") or default_runs_dir),
        name=str(d.get("name") or ""),
        auto_increment_name=bool(d.get("auto_increment_name", True)),
        params={**default_values(), **params},
    )


def _task_from_model_name(model_name: str) -> TaskKind:
    from .models_registry import ModelsRegistry
    if model_name and ModelsRegistry.is_known(model_name):
        return ModelsRegistry.infer_task_from_name(model_name)
    return TaskKind.DETECTOR
