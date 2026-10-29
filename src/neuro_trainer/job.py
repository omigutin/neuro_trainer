from __future__ import annotations
"""
Задание на обучение/проверку, которое выполняется в отдельном процессе (job_worker),
и его статус. Оба хранятся JSON-файлами в папке рана.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from .tasks import TaskKind
from .trainer import Trainer

# Имена служебных файлов задания в папке рана
SPEC_FILE = "job_spec.json"
STATUS_FILE = "job_status.json"
LOG_FILE = "train.log"


class JobKind(str, Enum):
    """ Что делает задание. """
    TRAIN = "train"
    VALIDATE = "validate"


class JobState(str, Enum):
    """ Состояние задания. """
    STARTING = "starting"
    RUNNING = "running"
    FINISHED = "finished"
    # Завершено досрочно по кнопке «Стоп» (после текущей эпохи)
    STOPPED = "stopped"
    # Процесс прерван принудительно
    KILLED = "killed"
    FAILED = "failed"

    @property
    def active(self) -> bool:
        """ Задание ещё выполняется. """
        return self in (JobState.STARTING, JobState.RUNNING)


@dataclass(slots=True)
class JobSpec:
    """ Всё, что нужно процессу обучения. """
    kind: JobKind
    task: TaskKind
    data: str
    project_dir: str
    run_name: str
    # Стандартные веса по имени (скачиваются в weights_dir) — либо model_path
    model_name: str = ""
    # Свои веса (.pt); при resume — last.pt прерванного рана
    model_path: str = ""
    # Кэш стандартных весов (configs/weights)
    weights_dir: str = ""
    # Аргументы model.train()/val() из каталога параметров
    train_args: dict[str, Any] = field(default_factory=dict)
    resume: bool = False

    def trainer(self, weights: Optional[str] = None) -> Trainer:
        """ Trainer для задания; weights — локальный путь к весам (по умолчанию model_path). """
        return Trainer(weights=weights or self.model_path, task=self.task, data=self.data,
                       project_dir=self.project_dir, run_name=self.run_name,
                       train_args=dict(self.train_args), resume=self.resume)

    @property
    def run_dir(self) -> Path:
        """ Папка рана (вычисляется без загрузки модели). """
        return self.trainer(weights=self.model_path or "unused.pt").save_dir

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        d["task"] = self.task.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "JobSpec":
        return cls(**{**d, "kind": JobKind(d["kind"]), "task": TaskKind(d["task"])})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "JobSpec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(slots=True)
class JobStatus:
    """ Статус задания (job_status.json). """
    state: JobState
    kind: JobKind
    started: str = ""
    finished: str = ""
    # Текст ошибки для FAILED
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state.value, "kind": self.kind.value, "started": self.started,
                "finished": self.finished, "error": self.error}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "JobStatus":
        return cls(state=JobState(d["state"]), kind=JobKind(d["kind"]), started=d.get("started", ""),
                   finished=d.get("finished", ""), error=d.get("error", ""))

    def save(self, run_dir: Path) -> None:
        path = Path(run_dir) / STATUS_FILE
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, run_dir: Path) -> Optional["JobStatus"]:
        path = Path(run_dir) / STATUS_FILE
        if not path.is_file():
            return None
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
