from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

from topbench.task_registry import CANONICAL_TASKS


EXCLUDE_SUFFIXES = (".log", ".txt")
EXCLUDE_NAMES = {".DS_Store", "stats_cache.json", "extracted_features.json"}
EXCLUDE_MARKERS = ("_eval", "_no_tool", "_with_tool", "_text_reasoning", "_agentic_workflow")


LEGACY_DATASET_ROOTS = {
    "single_point_prediction": Path("B1andB3") / "B1",
    "decision_making": Path("B2"),
    "treatment_effect_analysis": Path("B3"),
    "ranking_and_filtering": Path("B4"),
}


def should_copy_dataset_file(path: Path, *, task_name: str) -> bool:
    if path.name in EXCLUDE_NAMES:
        return False
    if path.suffix in EXCLUDE_SUFFIXES:
        return False
    if path.suffix == ".json" and any(marker in path.stem for marker in EXCLUDE_MARKERS):
        return False
    if task_name == "ranking_and_filtering" and path.suffix == ".json":
        if path.name not in {"info.json", "info_mod.json"} and "_current" not in path.stem:
            return False
    return path.suffix in {".json", ".csv"}


def iter_dataset_files(source_root: Path, *, task_name: str) -> Iterable[Path]:
    for path in source_root.rglob("*"):
        if path.is_file() and should_copy_dataset_file(path, task_name=task_name):
            yield path


def convert_legacy_layout(project_root: str | Path, output_data_root: str | Path) -> None:
    project = Path(project_root)
    output = Path(output_data_root)
    output.mkdir(parents=True, exist_ok=True)
    for canonical_name, rel_source in LEGACY_DATASET_ROOTS.items():
        source_root = project / rel_source
        if not source_root.exists():
            continue
        target_root = output / canonical_name
        for source_file in iter_dataset_files(source_root, task_name=canonical_name):
            rel = source_file.relative_to(source_root)
            target_file = target_root / rel
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target_file)
