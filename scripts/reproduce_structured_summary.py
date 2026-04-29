from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from topbench.compatibility.legacy_data_view import prepare_legacy_data_view
from topbench.compatibility.legacy_name_mapping import to_legacy_mode
from topbench.compatibility.legacy_reproduction import bundled_legacy_code_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay B4 structured evaluation in a temporary output root.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--frozen-output-root", default="outputs")
    parser.add_argument("--output-root", default="/tmp/topbench_release_b4_summary_replay")
    parser.add_argument("--model", default="deepseek")
    parser.add_argument("--mode", default="agentic_workflow")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--compare", action="store_true")
    return parser.parse_args()

def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    legacy_mode = to_legacy_mode(args.mode)
    output_root = Path(args.output_root)
    temp_inference_root = output_root / "inference_root"
    temp_dataset_root = output_root / "legacy_b4_dataset"
    output_root.mkdir(parents=True, exist_ok=True)

    frozen_b4 = Path(args.frozen_output_root) / args.model / legacy_mode / "B4"
    if not frozen_b4.exists():
        raise FileNotFoundError(f"Missing frozen B4 output dir: {frozen_b4}")

    temp_model_mode = temp_inference_root / args.model / legacy_mode
    temp_model_mode.mkdir(parents=True, exist_ok=True)
    (temp_model_mode / "B4").parent.mkdir(parents=True, exist_ok=True)
    if not (temp_model_mode / "B4").exists():
        (temp_model_mode / "B4").symlink_to(frozen_b4.resolve(), target_is_directory=True)

    prepared = prepare_legacy_data_view(args.data_root, temp_dataset_root, tasks=["ranking_and_filtering"])

    script = bundled_legacy_code_root() / "eval_b4_v3.py"
    if not script.exists():
        script = bundled_legacy_code_root() / "eval_v4_v2.py"
    cmd = [
        args.python_bin,
        str(script),
        "--inference_root",
        str(temp_inference_root),
        "--dataset_root",
        str(prepared["ranking_and_filtering"]),
        "--models",
        args.model,
        "--mode",
        legacy_mode,
        "--workers",
        str(args.workers),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    replay_summary = temp_inference_root / args.model / legacy_mode / "B4_summary.json"
    report = {
        "returncode": result.returncode,
        "replay_summary": str(replay_summary),
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }
    if args.compare:
        saved_summary = Path(args.frozen_output_root) / args.model / legacy_mode / "B4_summary.json"
        replayed = load_json(replay_summary)
        saved = load_json(saved_summary)
        report["comparison"] = {
            "saved_summary": str(saved_summary),
            "status": "match" if replayed == saved else "different",
            "replayed": replayed,
            "saved": saved,
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
