from __future__ import annotations

import argparse
import json

from topbench.evaluation.reasoning_evaluator import summarize_existing_reasoning_evals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize existing TopBench reasoning-task eval JSON files.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", default="text_reasoning")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--write-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_existing_reasoning_evals(
        output_root=args.output_root,
        model=args.model,
        mode=args.mode,
        task_name=args.task,
        write_path=args.write_path,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
