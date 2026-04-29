from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


@dataclass(frozen=True)
class TaskSpec:
    canonical_name: str
    legacy_name: str
    display_name: str
    requires_current_table: bool
    output_modality: str


CANONICAL_TASKS: Dict[str, TaskSpec] = {
    "single_point_prediction": TaskSpec(
        canonical_name="single_point_prediction",
        legacy_name="B1",
        display_name="Single-Point Prediction",
        requires_current_table=False,
        output_modality="natural_language",
    ),
    "decision_making": TaskSpec(
        canonical_name="decision_making",
        legacy_name="B2",
        display_name="Decision Making",
        requires_current_table=False,
        output_modality="natural_language",
    ),
    "treatment_effect_analysis": TaskSpec(
        canonical_name="treatment_effect_analysis",
        legacy_name="B3",
        display_name="Treatment Effect Analysis",
        requires_current_table=False,
        output_modality="natural_language",
    ),
    "ranking_and_filtering": TaskSpec(
        canonical_name="ranking_and_filtering",
        legacy_name="B4",
        display_name="Ranking and Filtering",
        requires_current_table=True,
        output_modality="structured_csv",
    ),
}

LEGACY_TO_CANONICAL = {spec.legacy_name: name for name, spec in CANONICAL_TASKS.items()}

TASK_ALIASES = {
    "single": "single_point_prediction",
    "single_point": "single_point_prediction",
    "single_point_prediction": "single_point_prediction",
    "b1": "single_point_prediction",
    "B1": "single_point_prediction",
    "decision": "decision_making",
    "decision_making": "decision_making",
    "b2": "decision_making",
    "B2": "decision_making",
    "treatment": "treatment_effect_analysis",
    "treatment_effect": "treatment_effect_analysis",
    "treatment_effect_analysis": "treatment_effect_analysis",
    "b3": "treatment_effect_analysis",
    "B3": "treatment_effect_analysis",
    "ranking": "ranking_and_filtering",
    "filtering": "ranking_and_filtering",
    "ranking_and_filtering": "ranking_and_filtering",
    "b4": "ranking_and_filtering",
    "B4": "ranking_and_filtering",
}


def canonicalize_task_name(task_name: str) -> str:
    try:
        canonical = TASK_ALIASES[task_name]
    except KeyError as exc:
        valid = ", ".join(sorted(CANONICAL_TASKS))
        raise ValueError(f"Unknown TopBench task '{task_name}'. Valid tasks: {valid}") from exc
    return canonical


def legacy_task_name(task_name: str) -> str:
    canonical = canonicalize_task_name(task_name)
    return CANONICAL_TASKS[canonical].legacy_name


def task_dir_aliases(task_name: str) -> Tuple[str, str]:
    canonical = canonicalize_task_name(task_name)
    return canonical, CANONICAL_TASKS[canonical].legacy_name


def iter_task_specs(task_names: Iterable[str] | None = None) -> Iterable[TaskSpec]:
    names = task_names or CANONICAL_TASKS.keys()
    for name in names:
        yield CANONICAL_TASKS[canonicalize_task_name(name)]
