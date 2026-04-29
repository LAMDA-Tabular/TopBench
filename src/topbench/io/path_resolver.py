from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from topbench.constants import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT
from topbench.task_registry import canonicalize_task_name, legacy_task_name


@dataclass(frozen=True)
class TopBenchPaths:
    data_root: Path = Path(DEFAULT_DATA_ROOT)
    output_root: Path = Path(DEFAULT_OUTPUT_ROOT)

    @classmethod
    def from_strings(cls, data_root: str | Path, output_root: str | Path) -> "TopBenchPaths":
        return cls(data_root=Path(data_root), output_root=Path(output_root))

    def dataset_task_dir(self, task_name: str) -> Path:
        return self.data_root / canonicalize_task_name(task_name)

    def legacy_dataset_task_dir(self, task_name: str) -> Path:
        return self.data_root / legacy_task_name(task_name)

    def model_mode_output_dir(self, model_name: str, mode: str, task_name: str) -> Path:
        return self.output_root / model_name / mode / canonicalize_task_name(task_name)

    def legacy_model_mode_output_dir(self, model_name: str, mode: str, task_name: str) -> Path:
        return self.output_root / model_name / mode / legacy_task_name(task_name)


def resolve_existing_task_dir(root: str | Path, task_name: str) -> Path:
    root_path = Path(root)
    canonical = canonicalize_task_name(task_name)
    candidates = [root_path / canonical, root_path / legacy_task_name(task_name)]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find task directory for '{task_name}' under {root_path}")
