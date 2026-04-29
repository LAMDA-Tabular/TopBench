from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from topbench.compatibility.legacy_name_mapping import to_legacy_mode, to_legacy_task_name
from topbench.evaluation.summary_builder import basic_stats
from topbench.task_registry import CANONICAL_TASKS, canonicalize_task_name, task_dir_aliases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay B1-B3 summaries from existing *_eval.json files without modifying frozen outputs.")
    parser.add_argument("--frozen-output-root", default="outputs")
    parser.add_argument("--output-root", default="/tmp/topbench_release_reasoning_summary_replay")
    parser.add_argument("--model", default="deepseek")
    parser.add_argument("--mode", default="text_reasoning")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["single_point_prediction", "decision_making", "treatment_effect_analysis"],
    )
    parser.add_argument("--compare", action="store_true")
    return parser.parse_args()


def nested_score(breakdown: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        item = breakdown.get(key)
        if isinstance(item, dict) and "score" in item:
            try:
                return float(item["score"])
            except (TypeError, ValueError):
                return None
        if isinstance(item, (int, float)):
            return float(item)
    return None


def extract_scores(eval_json: Dict[str, Any], task: str) -> Dict[str, Optional[float]]:
    breakdown = eval_json.get("breakdown", {}) or {}
    if task == "single_point_prediction":
        acc_keys = ["accuracy"]
        decision_keys: List[str] = []
    elif task == "decision_making":
        acc_keys = ["avg_prediction", "prediction", "accuracy"]
        decision_keys = ["decision", "selection"]
    else:
        acc_keys = ["pred_002_accuracy", "value_accuracy", "accuracy"]
        decision_keys = ["trend_accuracy"]

    return {
        "final": float(eval_json.get("final_score", 0.0) or 0.0),
        "accuracy": nested_score(breakdown, acc_keys),
        "logic": nested_score(breakdown, ["logic"]),
        "decision": nested_score(breakdown, decision_keys),
    }


def collect_eval_files(frozen_output_root: Path, *, model: str, legacy_mode: str, task: str) -> List[Path]:
    files: List[Path] = []
    for dirname in task_dir_aliases(task):
        root = frozen_output_root / model / legacy_mode / dirname
        if root.exists():
            files.extend(sorted(root.rglob("*_eval.json")))
    return sorted(set(files))


def collect_inference_files(frozen_output_root: Path, *, model: str, legacy_mode: str, task: str) -> List[Path]:
    files: List[Path] = []
    for dirname in task_dir_aliases(task):
        root = frozen_output_root / model / legacy_mode / dirname
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            name = path.name
            if "_eval" in name:
                continue
            if "tool" not in name:
                continue
            files.append(path)
    return sorted(set(files))


def eval_path_to_inference_path(path: Path) -> Path:
    name = path.name
    if name.endswith("_eval.json"):
        return path.with_name(name[:-10] + ".json")
    return path


def load_skipped_inference_paths(frozen_output_root: Path, *, legacy_task: str, legacy_mode: str, model: str) -> set[Path]:
    path = frozen_output_root / f"skipped_{legacy_task}_{legacy_mode}.csv"
    if not path.exists():
        return set()
    skipped: set[Path] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("model") and row["model"] != model:
                continue
            file_path = row.get("file_path")
            if file_path:
                skipped.add(Path(file_path))
    return skipped


def load_error_inference_paths(frozen_output_root: Path, *, legacy_task: str, legacy_mode: str, model: str) -> set[Path]:
    path = frozen_output_root / model / legacy_mode / f"errors_{legacy_task}_{model}_{legacy_mode}.csv"
    if not path.exists():
        return set()
    errored: set[Path] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_path = row.get("file_path")
            if file_path:
                errored.add(Path(file_path))
    return errored


def replay_task_summary(frozen_output_root: Path, *, model: str, legacy_mode: str, task: str) -> Dict[str, Any]:
    legacy_task = to_legacy_task_name(task)
    eval_files = collect_eval_files(frozen_output_root, model=model, legacy_mode=legacy_mode, task=task)
    skipped_paths = load_skipped_inference_paths(
        frozen_output_root,
        legacy_task=legacy_task,
        legacy_mode=legacy_mode,
        model=model,
    )
    error_paths = load_error_inference_paths(
        frozen_output_root,
        legacy_task=legacy_task,
        legacy_mode=legacy_mode,
        model=model,
    )
    ignored_paths = skipped_paths | error_paths
    filtered_eval_files = [
        path for path in eval_files if eval_path_to_inference_path(path) not in ignored_paths
    ]
    final_scores: List[float] = []
    accuracy_scores: List[float] = []
    logic_scores: List[float] = []
    decision_scores: List[float] = []
    empty_count = 0

    for path in filtered_eval_files:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        judge_output = data.get("judge_output", {}) or {}
        if isinstance(judge_output, dict) and judge_output.get("error") == "Empty Response":
            empty_count += 1
        scores = extract_scores(data, task)
        final_scores.append(float(scores["final"] or 0.0))
        if scores["accuracy"] is not None:
            accuracy_scores.append(float(scores["accuracy"]))
        if scores["logic"] is not None:
            logic_scores.append(float(scores["logic"]))
        if scores["decision"] is not None:
            decision_scores.append(float(scores["decision"]))

    skipped_count = len(skipped_paths)
    failed_count = len(error_paths)
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "mode": legacy_mode,
        "benchmark": legacy_task,
        "counts": {
            "total": len(final_scores) + skipped_count + failed_count,
            "success": len(final_scores) - empty_count,
            "empty": empty_count,
            "failed": failed_count,
            "skipped": skipped_count,
        },
        "stats": {
            "final_score": basic_stats(final_scores),
            "accuracy_score": basic_stats(accuracy_scores),
            "logic_score": basic_stats(logic_scores),
            "decision_score": basic_stats(decision_scores) if decision_scores else None,
        },
    }


def load_saved_summary(frozen_output_root: Path, *, model: str, legacy_mode: str, task: str) -> Dict[str, Any] | None:
    legacy_task = to_legacy_task_name(task)
    candidates = [
        frozen_output_root / model / legacy_mode / f"summary_{legacy_task}_{model}_{legacy_mode}.json",
        frozen_output_root / model / legacy_mode / f"summary_{task}_{model}_{legacy_mode}.json",
    ]
    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            data["_summary_path"] = str(path)
            return data
    return None


def compare_summary(replayed: Dict[str, Any], saved: Dict[str, Any] | None) -> Dict[str, Any]:
    if saved is None:
        return {"status": "missing_saved_summary"}
    diffs = {}
    for section in ["counts", "stats"]:
        if replayed.get(section) != saved.get(section):
            diffs[section] = {"replayed": replayed.get(section), "saved": saved.get(section)}
    return {
        "status": "match" if not diffs else "different",
        "saved_summary": saved.get("_summary_path"),
        "diffs": diffs,
    }


def main() -> None:
    args = parse_args()
    frozen_output_root = Path(args.frozen_output_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    legacy_mode = to_legacy_mode(args.mode)

    reports = []
    for raw_task in args.tasks:
        task = canonicalize_task_name(raw_task)
        if task == "ranking_and_filtering":
            continue
        replayed = replay_task_summary(frozen_output_root, model=args.model, legacy_mode=legacy_mode, task=task)
        out_path = output_root / args.model / legacy_mode / f"summary_{to_legacy_task_name(task)}_{args.model}_{legacy_mode}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(replayed, f, ensure_ascii=False, indent=2)
        item = {"task": task, "replayed_summary": str(out_path), "summary": replayed}
        if args.compare:
            saved = load_saved_summary(frozen_output_root, model=args.model, legacy_mode=legacy_mode, task=task)
            item["comparison"] = compare_summary(replayed, saved)
        reports.append(item)

    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
