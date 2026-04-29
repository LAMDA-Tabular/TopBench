from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download TopBench data from HuggingFace after release.")
    parser.add_argument("--repo-id", required=True, help="HuggingFace dataset repo id, e.g. LAMDA-Tabular/TopBench")
    parser.add_argument("--local-dir", default="data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        sys.executable,
        "-m",
        "huggingface_hub",
        "download",
        args.repo_id,
        "--repo-type",
        "dataset",
        "--local-dir",
        args.local_dir,
    ]
    raise SystemExit(subprocess.run(cmd, check=False).returncode)


if __name__ == "__main__":
    main()
