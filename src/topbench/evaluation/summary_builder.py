from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def basic_stats(values: Iterable[float]) -> Dict[str, float]:
    items = [float(v) for v in values]
    if not items:
        return {"avg": 0.0, "max": 0.0, "min": 0.0}
    return {
        "avg": round(sum(items) / len(items), 4),
        "max": round(max(items), 4),
        "min": round(min(items), 4),
    }


def build_summary(records: List[Dict[str, Any]], *, model: str, mode: str, task_name: str) -> Dict[str, Any]:
    scores = [float(row["final_score"]) for row in records if "final_score" in row]
    return {
        "model": model,
        "mode": mode,
        "task": task_name,
        "counts": {
            "total": len(records),
            "success": len(scores),
            "failed": len(records) - len(scores),
        },
        "stats": {
            "final_score": basic_stats(scores),
        },
    }


def write_summary(summary: Dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
