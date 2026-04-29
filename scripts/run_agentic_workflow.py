from __future__ import annotations

import argparse

from topbench.inference.agentic_workflow_runner import run_agentic_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TopBench agentic workflow inference.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", default="deepseek")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--sandbox-image", default="topbench-sandbox:latest")
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = run_agentic_workflow(
        task_name=args.task,
        model_name=args.model,
        data_root=args.data_root,
        output_root=args.output_root,
        sandbox_image=args.sandbox_image,
        max_files=args.max_files,
    )
    print(f"Wrote {count} agentic-workflow outputs.")


if __name__ == "__main__":
    main()
