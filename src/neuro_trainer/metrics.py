from __future__ import annotations
"""
MetricsStore — финальные метрики рядом с results.csv.
Использование в Trainer: MetricsStore.save_final_metrics(run_dir)
Реализация «best-effort»: если results.csv не найден, пишем пустые структуры.
"""

from pathlib import Path
from typing import Dict, Any
import csv, json


class MetricsStore:
    @staticmethod
    def _read_results_csv(run_dir: Path) -> Dict[str, Any]:
        csv_path = run_dir / "results.csv"
        if not csv_path.exists():
            return {}
        # берём последнюю строку (обычно — итоговая)
        last_row: Dict[str, str] | None = None
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                last_row = row
        return last_row or {}

    @staticmethod
    def save_final_metrics(run_dir: Path | str) -> Dict[str, Any]:
        d = Path(run_dir)
        d.mkdir(parents=True, exist_ok=True)

        row = MetricsStore._read_results_csv(d)

        # простая нормализация типов
        payload: Dict[str, Any] = {}
        for k, v in row.items():
            try:
                vv = v.strip()
                if vv == "":
                    payload[k] = v
                else:
                    payload[k] = float(v) if "." in vv or "e" in vv.lower() else int(v)
            except Exception:
                payload[k] = v

        (d / "final_metrics.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (d / "final_metrics_extra.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for k, v in payload.items():
                w.writerow([k, v])
        return payload
