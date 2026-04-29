from __future__ import annotations

import argparse
import sys

from topbench.compatibility.legacy_reproduction import run_legacy_predict_only_baseline
from topbench.task_registry import canonicalize_task_name

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TopBench predict-only ensemble baseline.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--mode", default="predict_only")
    parser.add_argument("--model-name", default="predict_only_ensemble")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--accuracy-first", action="store_true")
    parser.add_argument("--full-complex-models", action="store_true")
    parser.add_argument("--gpu", default="off", choices=["off", "auto", "cuda"])
    parser.add_argument("--tabpfn-max-rows", type=int, default=None)
    parser.add_argument("--tabstar-max-rows", type=int, default=None)
    parser.add_argument("--tabstar-time-limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--fast-smoke",
        action="store_true",
        help="Limit the smoke run to a very small subset with conservative row caps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = canonicalize_task_name(args.task)
    max_train_rows = args.max_train_rows
    max_files = args.max_files
    if args.fast_smoke and max_files is None:
        max_files = 1
    if args.fast_smoke and max_train_rows is None:
        max_train_rows = 5000

    result = run_legacy_predict_only_baseline(
        data_root=args.data_root,
        output_root=args.output_root,
        task_names=[task],
        model_name=args.model_name,
        mode=args.mode,
        python_bin=args.python_bin,
        skip_existing=args.skip_existing,
        max_files=max_files,
        accuracy_first=args.accuracy_first,
        full_complex_models=args.full_complex_models,
        gpu=args.gpu,
        max_train_rows=max_train_rows,
        tabpfn_max_rows=args.tabpfn_max_rows,
        tabstar_max_rows=args.tabstar_max_rows,
        tabstar_time_limit=args.tabstar_time_limit,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
