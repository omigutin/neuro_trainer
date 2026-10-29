from __future__ import annotations
"""
JobManager — запуск обучения в отдельном процессе и наблюдение за ним.

Одновременно выполняется одно задание (одна видеокарта). Веб-сервер не блокируется,
память видеокарты освобождается вместе с процессом.
"""

import json
import os
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .callbacks import StopFlag
from .job import LOG_FILE, SPEC_FILE, JobSpec, JobState, JobStatus, now_iso
from .trainer import PROGRESS_FILE, STOP_FILE

# Каталог src (для PYTHONPATH процесса обучения, если пакет не установлен)
_SRC_DIR = Path(__file__).resolve().parent.parent
# Сколько последних байт лога отдавать в интерфейс
LOG_TAIL_BYTES = 64 * 1024


@dataclass(slots=True)
class JobView:
    """ Снимок задания для интерфейса. """
    state: JobState
    kind: str
    run_dir: str
    task: str
    started: str
    finished: str
    error: str
    # Код возврата процесса (None — ещё работает)
    returncode: Optional[int]

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state.value, "active": self.state.active, "kind": self.kind,
                "run_dir": self.run_dir, "task": self.task, "started": self.started,
                "finished": self.finished, "error": self.error, "returncode": self.returncode}


class JobBusyError(RuntimeError):
    """ Уже выполняется другое задание. """


class JobManager:
    """ Одно задание за раз, в отдельном процессе. """

    def __init__(self, python: str = sys.executable) -> None:
        # Интерпретатор для процесса обучения
        self._python = python
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._spec: Optional[JobSpec] = None
        self._run_dir: Optional[Path] = None
        self._killed = False

    # ---------- запуск / остановка ----------

    def start(self, spec: JobSpec) -> JobView:
        """ Запустить задание. JobBusyError, если предыдущее ещё выполняется. """
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                raise JobBusyError("Another job is already running")
            run_dir = spec.run_dir
            run_dir.mkdir(parents=True, exist_ok=True)
            spec_path = run_dir / SPEC_FILE
            spec.save(spec_path)
            StopFlag(run_dir / STOP_FILE).reset()
            JobStatus(state=JobState.STARTING, kind=spec.kind, started=now_iso()).save(run_dir)

            cwd = Path(spec.weights_dir) if spec.weights_dir else run_dir
            cwd.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONPATH"] = os.pathsep.join(p for p in (str(_SRC_DIR), env.get("PYTHONPATH", "")) if p)

            mode = "ab" if spec.resume else "wb"
            with (run_dir / LOG_FILE).open(mode) as log:
                self._proc = subprocess.Popen(
                    [self._python, "-m", "neuro_trainer.job_worker", str(spec_path)],
                    cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    **_process_group_kwargs(),
                )
            self._spec = spec
            self._run_dir = run_dir
            self._killed = False
            return self._view()

    def request_stop(self) -> None:
        """ Остановить после текущей эпохи (с сохранением весов). """
        with self._lock:
            if self._run_dir is None or not self._is_alive():
                raise RuntimeError("No running job")
            StopFlag(self._run_dir / STOP_FILE).request()

    def kill(self) -> None:
        """ Прервать процесс немедленно (вместе с процессами загрузки данных). """
        with self._lock:
            if self._proc is None or not self._is_alive():
                raise RuntimeError("No running job")
            _kill_tree(self._proc)
            self._proc.wait(timeout=30)
            self._killed = True
            if self._run_dir is not None and self._spec is not None:
                status = JobStatus.load(self._run_dir) or JobStatus(state=JobState.RUNNING, kind=self._spec.kind)
                status.state = JobState.KILLED
                status.finished = now_iso()
                status.save(self._run_dir)

    # ---------- наблюдение ----------

    def status(self) -> Optional[JobView]:
        """ Текущее (или последнее) задание; None — заданий не было. """
        with self._lock:
            return self._view() if self._spec is not None else None

    def progress(self, since: int = 0) -> list[dict[str, Any]]:
        """ Строки метрик по эпохам начиная с индекса since. """
        if self._run_dir is None:
            return []
        path = self._run_dir / PROGRESS_FILE
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < since or not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # последняя строка может быть дописана не полностью
                    break
        return rows

    def log_tail(self, max_bytes: int = LOG_TAIL_BYTES) -> str:
        """ Конец лога процесса обучения. Прогресс-бары (\\r) сворачиваются до последнего состояния. """
        if self._run_dir is None:
            return ""
        path = self._run_dir / LOG_FILE
        if not path.is_file():
            return ""
        size = path.stat().st_size
        with path.open("rb") as f:
            f.seek(max(0, size - max_bytes))
            text = f.read().decode("utf-8", errors="replace")
        lines = [line.rsplit("\r", 1)[-1] for line in text.replace("\r\n", "\n").split("\n")]
        return "\n".join(lines)

    # ---------- внутреннее ----------

    def _is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _view(self) -> JobView:
        assert self._spec is not None and self._run_dir is not None
        status = JobStatus.load(self._run_dir)
        returncode = self._proc.poll() if self._proc is not None else None
        if status is None:
            status = JobStatus(state=JobState.STARTING, kind=self._spec.kind)
        state = status.state
        error = status.error
        # Процесс завершился, не записав итог (упал до записи статуса или убит извне)
        if returncode is not None and state.active:
            state = JobState.KILLED if self._killed else JobState.FAILED
            error = error or f"Process exited with code {returncode}, see {LOG_FILE}"
        return JobView(state=state, kind=self._spec.kind.value, run_dir=str(self._run_dir),
                       task=self._spec.task.value, started=status.started, finished=status.finished,
                       error=error, returncode=returncode)


def _process_group_kwargs() -> dict[str, Any]:
    """ Отдельная группа процессов, чтобы прерывать обучение вместе с процессами загрузки данных. """
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _kill_tree(proc: subprocess.Popen[bytes]) -> None:
    """ Завершить процесс и всех его потомков. """
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, check=False)
    else:
        os.killpg(proc.pid, signal.SIGKILL)
