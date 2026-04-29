from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from topbench.io.dataset_loader import save_json


def output_file_for_query(
    output_root: str | Path,
    *,
    model_name: str,
    mode: str,
    task_name: str,
    query_file: Path,
) -> Path:
    return Path(output_root) / model_name / mode / task_name / f"{query_file.stem}_{model_name}_{mode}.json"


def write_prediction_output(output_path: Path, payload: Dict[str, Any]) -> None:
    save_json(payload, output_path)
