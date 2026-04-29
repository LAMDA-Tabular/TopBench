from __future__ import annotations

from topbench.task_registry import CANONICAL_TASKS, LEGACY_TO_CANONICAL, canonicalize_task_name


def to_legacy_task_name(task_name: str) -> str:
    return CANONICAL_TASKS[canonicalize_task_name(task_name)].legacy_name


def from_legacy_task_name(task_name: str) -> str:
    return LEGACY_TO_CANONICAL.get(task_name, canonicalize_task_name(task_name))


def to_legacy_mode(mode: str) -> str:
    mapping = {
        "text_reasoning": "no_tool",
        "agentic_workflow": "with_tool",
        "predict_only": "with_tool",
        "no_tool": "no_tool",
        "with_tool": "with_tool",
    }
    if mode not in mapping:
        raise ValueError(f"Unknown mode '{mode}'")
    return mapping[mode]
