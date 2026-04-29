from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TopBench text-based reasoning inference.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", default="deepseek")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--max-table-tokens", type=int, default=24000)
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from topbench.inference.text_reasoning_runner import run_text_reasoning

    count = run_text_reasoning(
        task_name=args.task,
        model_name=args.model,
        data_root=args.data_root,
        output_root=args.output_root,
        max_table_tokens=args.max_table_tokens,
        max_files=args.max_files,
    )
    print(f"Wrote {count} text-reasoning outputs.")


if __name__ == "__main__":
    main()
