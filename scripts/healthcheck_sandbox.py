from __future__ import annotations

import argparse
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether the TopBench sandbox image can run Python.")
    parser.add_argument("--image", default="topbench-sandbox:latest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        args.image,
        "python",
        "-c",
        "import pandas, sklearn; print('TopBench sandbox OK')",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
