from __future__ import annotations

from pathlib import Path

from topbench.evaluation.summary_builder import build_summary, write_summary
from topbench.io.dataset_loader import load_json
from topbench.task_registry import canonicalize_task_name, task_dir_aliases


def collect_existing_eval_records(output_root: str | Path, *, model: str, mode: str, task_name: str) -> list[dict]:
    task = canonicalize_task_name(task_name)
    records: list[dict] = []
    for task_dir_name in task_dir_aliases(task):
        root = Path(output_root) / model / mode / task_dir_name
        for path in sorted(root.rglob("*_eval.json")) if root.exists() else []:
            data = load_json(path)
            data["_eval_path"] = str(path)
            records.append(data)
    return records


def summarize_existing_reasoning_evals(
    *,
    output_root: str | Path,
    model: str,
    mode: str,
    task_name: str,
    write_path: str | Path | None = None,
) -> dict:
    task = canonicalize_task_name(task_name)
    records = collect_existing_eval_records(output_root, model=model, mode=mode, task_name=task)
    summary = build_summary(records, model=model, mode=mode, task_name=task)
    if write_path is not None:
        write_summary(summary, write_path)
    return summary
