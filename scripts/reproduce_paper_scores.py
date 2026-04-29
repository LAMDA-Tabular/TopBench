from __future__ import annotations

import argparse
import sys

from topbench.compatibility.legacy_reproduction import run_legacy_reasoning_evaluator, run_legacy_structured_evaluator
from topbench.task_registry import canonicalize_task_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce paper-score evaluation from frozen outputs using legacy-compatible evaluators.")
    parser.add_argument(
        "--legacy-code-root",
        default=None,
        help="Directory containing legacy eval scripts. Defaults to the bundled compatibility scripts.",
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--inference-root", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", default="text_reasoning")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--rerun-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = canonicalize_task_name(args.task)
    if task == "ranking_and_filtering":
        result = run_legacy_structured_evaluator(
            legacy_code_root=args.legacy_code_root,
            inference_root=args.inference_root,
            data_root=args.data_root,
            model_name=args.model,
            mode=args.mode,
            python_bin=args.python_bin,
        )
    else:
        result = run_legacy_reasoning_evaluator(
            legacy_code_root=args.legacy_code_root,
            inference_root=args.inference_root,
            data_root=args.data_root,
            task_name=task,
            model_name=args.model,
            mode=args.mode,
            workers=args.workers,
            python_bin=args.python_bin,
            skip_existing=not args.rerun_existing,
        )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
