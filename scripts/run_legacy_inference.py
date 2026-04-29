from __future__ import annotations

import argparse
import json
import sys

from topbench.compatibility.legacy_reproduction import run_legacy_inference
from topbench.task_registry import CANONICAL_TASKS, canonicalize_task_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run legacy-compatible TopBench inference so the output JSON/CSV schema matches the paper-era scripts."
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--model", default="deepseek")
    parser.add_argument("--tasks", nargs="+", default=list(CANONICAL_TASKS.keys()))
    parser.add_argument("--modes", nargs="+", default=["text_reasoning", "agentic_workflow"])
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--truncation-strategy", default=None)
    parser.add_argument("--no-skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = []
    for raw_task in args.tasks:
        task = canonicalize_task_name(raw_task)
        for mode in args.modes:
            result = run_legacy_inference(
                data_root=args.data_root,
                output_root=args.output_root,
                task_name=task,
                model_name=args.model,
                mode=mode,
                workers=args.workers,
                python_bin=args.python_bin,
                max_files=args.max_files,
                skip_existing=not args.no_skip_existing,
                truncation_strategy=args.truncation_strategy,
            )
            report.append(
                {
                    "task": task,
                    "mode": mode,
                    "returncode": result.returncode,
                    "stdout_tail": result.stdout[-1000:],
                    "stderr_tail": result.stderr[-1000:],
                }
            )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if any(item["returncode"] != 0 for item in report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
