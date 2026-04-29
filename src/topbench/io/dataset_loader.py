from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List

from topbench.constants import CURRENT_CSV, HISTORY_CSV, INFO_JSON, INFO_MOD_JSON
from topbench.io.path_resolver import resolve_existing_task_dir
from topbench.task_registry import canonicalize_task_name


@dataclass(frozen=True)
class DatasetFolder:
    task_name: str
    path: Path
    history_csv: Path
    current_csv: Path | None
    info_json: Path | None
    query_files: List[Path]


GENERATED_MARKERS = (
    "_no_tool",
    "_with_tool",
    "_text_reasoning",
    "_agentic_workflow",
    "_eval",
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Path, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def is_query_json(path: Path, *, task_name: str | None = None) -> bool:
    if path.name in {INFO_JSON, INFO_MOD_JSON, "stats_cache.json", "extracted_features.json"}:
        return False
    if not path.name.endswith(".json"):
        return False
    if task_name == "ranking_and_filtering" and "_current" not in path.stem:
        return False
    return not any(marker in path.stem for marker in GENERATED_MARKERS)


def iter_dataset_folders(data_root: str | Path, task_name: str) -> Iterator[DatasetFolder]:
    task = canonicalize_task_name(task_name)
    task_dir = resolve_existing_task_dir(data_root, task)
    for history_csv in sorted(task_dir.rglob(HISTORY_CSV)):
        dataset_dir = history_csv.parent
        query_files = sorted(path for path in dataset_dir.glob("*.json") if is_query_json(path, task_name=task))
        if not query_files:
            continue
        current_csv = dataset_dir / CURRENT_CSV
        info_json = dataset_dir / INFO_JSON
        yield DatasetFolder(
            task_name=task,
            path=dataset_dir,
            history_csv=history_csv,
            current_csv=current_csv if current_csv.exists() else None,
            info_json=info_json if info_json.exists() else None,
            query_files=query_files,
        )


def load_dataset_metadata(dataset_dir: str | Path) -> Dict[str, Any]:
    dataset_path = Path(dataset_dir)
    for name in [INFO_MOD_JSON, INFO_JSON]:
        candidate = dataset_path / name
        if candidate.exists():
            return load_json(candidate)
    return {}
