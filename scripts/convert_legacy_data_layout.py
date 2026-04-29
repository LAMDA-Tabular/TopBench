from __future__ import annotations

import argparse

from topbench.io.legacy_layout_converter import convert_legacy_layout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert legacy B1/B2/B3/B4 data folders to canonical TopBench task names.")
    parser.add_argument("--legacy-project-root", required=True)
    parser.add_argument("--output-data-root", default="data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_legacy_layout(args.legacy_project_root, args.output_data_root)


if __name__ == "__main__":
    main()
