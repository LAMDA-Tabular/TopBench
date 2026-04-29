from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from topbench.compatibility.legacy_name_mapping import to_legacy_mode, to_legacy_task_name
from topbench.compatibility.legacy_reproduction import run_legacy_inference
from topbench.task_registry import CANONICAL_TASKS, canonicalize_task_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run B1-B4 text/agentic inference smoke tests using the release compatibility runner."
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="/tmp/topbench_release_inference_smoke")
    parser.add_argument("--model", default="deepseek")
    parser.add_argument("--tasks", nargs="+", default=list(CANONICAL_TASKS.keys()))
    parser.add_argument("--modes", nargs="+", default=["text_reasoning", "agentic_workflow"])
    parser.add_argument("--max-files", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--reference-output-root", default=None, help="Optional frozen outputs root for schema comparison.")
    return parser.parse_args()


def find_one_json(root: Path) -> Path | None:
    for path in sorted(root.rglob("*.json")):
        if "_eval" not in path.name and "summary" not in path.name:
            return path
    return None


def compare_json_schema(new_file: Path, reference_root: Path, *, model: str, mode: str, task: str) -> dict:
    legacy_task = to_legacy_task_name(task)
    legacy_mode = to_legacy_mode(mode)
    ref_task_root = reference_root / model / legacy_mode / legacy_task
    reference = find_one_json(ref_task_root)
    if reference is None:
        return {"status": "no_reference_json", "reference_root": str(ref_task_root)}
    with new_file.open("r", encoding="utf-8") as f:
        new_data = json.load(f)
    with reference.open("r", encoding="utf-8") as f:
        ref_data = json.load(f)
    return {
        "status": "compared",
        "new_file": str(new_file),
        "reference_file": str(reference),
        "missing_keys_in_new": sorted(set(ref_data) - set(new_data)),
        "extra_keys_in_new": sorted(set(new_data) - set(ref_data)),
    }


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    report = []
    for raw_task in args.tasks:
        task = canonicalize_task_name(raw_task)
        task_data_root = data_root / task
        if not task_data_root.exists():
            report.append({"task": task, "status": "missing_data", "path": str(task_data_root)})
            continue
        for mode in args.modes:
            legacy_mode = to_legacy_mode(mode)
            result = run_legacy_inference(
                data_root=args.data_root,
                output_root=args.output_root,
                task_name=task,
                model_name=args.model,
                mode=mode,
                workers=args.workers,
                python_bin=args.python_bin,
                max_files=args.max_files,
                skip_existing=True,
            )
            task_out = output_root / args.model / legacy_mode / to_legacy_task_name(task)
            new_json = find_one_json(task_out)
            item = {
                "task": task,
                "mode": mode,
                "legacy_mode": legacy_mode,
                "returncode": result.returncode,
                "output_task_root": str(task_out),
                "generated_json": str(new_json) if new_json else None,
                "stdout_tail": result.stdout[-1000:],
                "stderr_tail": result.stderr[-1000:],
            }
            if new_json and args.reference_output_root:
                item["schema_compare"] = compare_json_schema(
                    new_json,
                    Path(args.reference_output_root),
                    model=args.model,
                    mode=mode,
                    task=task,
                )
            report.append(item)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if any(item.get("returncode", 0) != 0 for item in report if item.get("status") != "missing_data"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
