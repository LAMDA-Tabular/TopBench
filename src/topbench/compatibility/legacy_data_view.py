from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable

from topbench.io.path_resolver import resolve_existing_task_dir
from topbench.task_registry import canonicalize_task_name


LEGACY_RELATIVE_TASK_DIRS = {
    "single_point_prediction": Path("B1andB3") / "B1",
    "decision_making": Path("B2"),
    "treatment_effect_analysis": Path("B3"),
    "ranking_and_filtering": Path("B4"),
}


def safe_symlink_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    try:
        os.symlink(source.resolve(), target, target_is_directory=source.is_dir())
    except OSError:
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            for child in source.iterdir():
                safe_symlink_or_copy(child, target / child.name)
        else:
            target.write_bytes(source.read_bytes())


def prepare_legacy_data_view(
    data_root: str | Path,
    legacy_view_root: str | Path,
    *,
    tasks: Iterable[str] | None = None,
) -> Dict[str, Path]:
    source_root = Path(data_root).resolve()
    legacy_root = Path(legacy_view_root).resolve()
    prepared: Dict[str, Path] = {}

    for raw_task in tasks or LEGACY_RELATIVE_TASK_DIRS.keys():
        task = canonicalize_task_name(raw_task)
        source_dir = resolve_existing_task_dir(source_root, task)
        target_dir = legacy_root / LEGACY_RELATIVE_TASK_DIRS[task]
        safe_symlink_or_copy(source_dir, target_dir)
        prepared[task] = target_dir

    return prepared
