from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json, csv

class MetricsStore:
    """
    Сохраняет метрики Ultralytics в два файла рядом с results.csv:
      - final_metrics.json        — удобен для программного чтения
      - final_metrics_extra.csv   — удобно открыть в Excel
    Понимает различия задач: CLS vs DET/SEG.
    """
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save(self, metrics_obj: Any, task: str | None) -> Dict[str, Any]:
        task = (task or "").lower()
        try:
            if task == "classifier":
                payload: Dict[str, Any] = {
                    "top1": float(getattr(metrics_obj, "top1", 0.0)),
                    "top5": float(getattr(metrics_obj, "top5", 0.0)),
                    "fitness": float(getattr(metrics_obj, "fitness", 0.0)),
                    "speed": getattr(metrics_obj, "speed", {}),
                }
            else:
                box = getattr(metrics_obj, "box", None)
                if box is not None:
                    payload = {
                        "map50_95": float(getattr(box, "map", 0.0)),
                        "map50": float(getattr(box, "map50", 0.0)),
                        "map75": float(getattr(box, "map75", 0.0)),
                        "maps_per_class": list(map(float, getattr(box, "maps", []) or [])),
                        "speed": getattr(metrics_obj, "speed", {}),
                    }
                else:
                    payload = {"raw": getattr(metrics_obj, "__dict__", str(metrics_obj))}
        except Exception as e:
            payload = {"error": f"failed to extract metrics: {e}", "raw": str(metrics_obj)}

        (self.run_dir / "final_metrics.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        with (self.run_dir / "final_metrics_extra.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for k, v in payload.items():
                w.writerow([k, v])
        return payload
