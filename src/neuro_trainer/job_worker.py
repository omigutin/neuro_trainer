from __future__ import annotations
"""
Процесс обучения: python -m neuro_trainer.job_worker <путь к job_spec.json>

Запускается JobManager'ом; вывод (stdout/stderr) уходит в train.log папки рана,
итог — в job_status.json.
"""

import logging
import sys
import traceback
from pathlib import Path

from .job import JobKind, JobSpec, JobState, JobStatus, now_iso
from .services.weights_resolver import WeightsResolver
from .weights_cache import WeightsCache


def run(spec_path: Path) -> int:
    """ Выполнить задание; код возврата 0 — успех. """
    spec = JobSpec.load(spec_path)
    run_dir = spec_path.parent
    status = JobStatus(state=JobState.RUNNING, kind=spec.kind, started=now_iso())
    status.save(run_dir)
    try:
        weights = WeightsResolver(WeightsCache(Path(spec.weights_dir))).resolve(spec.model_name, spec.model_path)
        trainer = spec.trainer(weights=weights)
        if spec.kind is JobKind.TRAIN:
            trainer.train()
            status.state = JobState.STOPPED if trainer.stop_flag.is_set() else JobState.FINISHED
        else:
            trainer.validate_only()
            status.state = JobState.FINISHED
    except BaseException as e:  # noqa: BLE001 — статус FAILED должен записаться при любой ошибке, затем пробрасываем
        status.state = JobState.FAILED
        status.error = f"{type(e).__name__}: {e}"
        status.finished = now_iso()
        status.save(run_dir)
        traceback.print_exc()
        if isinstance(e, Exception):
            return 1
        raise
    status.finished = now_iso()
    status.save(run_dir)
    return 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m neuro_trainer.job_worker <job_spec.json>")
    sys.exit(run(Path(sys.argv[1])))


if __name__ == "__main__":
    main()
