from __future__ import annotations

import argparse
from pathlib import Path

from topbench.io.dataset_loader import iter_dataset_folders
from topbench.task_registry import CANONICAL_TASKS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a TopBench dataset layout.")
    parser.add_argument("--data-root", default="data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    total_queries = 0
    for task_name in CANONICAL_TASKS:
        folders = list(iter_dataset_folders(data_root, task_name))
        query_count = sum(len(folder.query_files) for folder in folders)
        total_queries += query_count
        print(f"{task_name}: {len(folders)} dataset folders, {query_count} queries")
    print(f"total_queries: {total_queries}")


if __name__ == "__main__":
    main()
