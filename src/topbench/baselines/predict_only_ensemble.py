from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from topbench.baselines.preprocessing import frame_from_feature_dicts, split_features_target
from topbench.baselines.tabular_model_pool import build_model_pool
from topbench.io.dataset_loader import iter_dataset_folders, load_dataset_metadata, load_json, save_json
from topbench.task_registry import canonicalize_task_name


def infer_target_column(metadata: Dict[str, Any]) -> str:
    target = metadata.get("target")
    if isinstance(target, dict) and target:
        return next(iter(target))
    if isinstance(target, str):
        return target
    raise ValueError("Cannot infer target column from info.json.")


def infer_task_type(metadata: Dict[str, Any], target_series: pd.Series) -> str:
    raw = str(metadata.get("task_type", "")).lower()
    if raw in {"classification", "regression"}:
        return raw
    if not pd.api.types.is_numeric_dtype(target_series) or target_series.nunique(dropna=True) <= 20:
        return "classification"
    return "regression"


def select_best_model(models: Dict[str, Any], x: pd.DataFrame, y: pd.Series, task_type: str) -> tuple[str, Any]:
    if len(x) < 30:
        name, model = next(iter(models.items()))
        model.fit(x, y)
        return name, model
    stratify = y if task_type == "classification" and y.nunique() > 1 else None
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=stratify
    )
    best_name = ""
    best_score = -np.inf
    best_model = None
    for name, model in models.items():
        try:
            model.fit(x_train, y_train)
            pred = model.predict(x_valid)
            score = accuracy_score(y_valid, pred) if task_type == "classification" else r2_score(y_valid, pred)
        except Exception:
            continue
        if score > best_score:
            best_name = name
            best_score = float(score)
            best_model = model
    if best_model is None:
        best_name, best_model = next(iter(models.items()))
        best_score = 0.0
    best_model.fit(x, y)
    return best_name or "fallback", best_model


def predict_structured_scenarios(model, feature_dicts: List[Dict[str, Any]]) -> List[Any]:
    frame = frame_from_feature_dicts(feature_dicts)
    return list(model.predict(frame))


def run_predict_only_ensemble(
    *,
    task_name: str,
    data_root: str | Path,
    output_root: str | Path,
    max_files: int | None = None,
    include_optional_models: bool = True,
    max_train_rows: int | None = None,
) -> int:
    task = canonicalize_task_name(task_name)
    output = Path(output_root)
    count = 0
    for dataset in iter_dataset_folders(data_root, task):
        metadata = load_dataset_metadata(dataset.path)
        history = pd.read_csv(dataset.history_csv, encoding_errors="replace", on_bad_lines="skip", low_memory=False)
        if max_train_rows is not None and len(history) > max_train_rows:
            history = history.sample(n=max_train_rows, random_state=42).reset_index(drop=True)
        target_col = infer_target_column(metadata)
        x, y = split_features_target(history, target_col)
        task_type = infer_task_type(metadata, y)
        models = build_model_pool(x, task_type, include_optional_models=include_optional_models)
        best_name, best_model = select_best_model(models, x, y, task_type)
        for query_file in dataset.query_files:
            query = load_json(query_file)
            scenarios = query.get("ground_truth", {}).get("extracted_features", [])
            feature_dicts = [item.get("features", {}) for item in scenarios]
            predictions = predict_structured_scenarios(best_model, feature_dicts) if feature_dicts else []
            result = {
                "model_name": "predict_only_ensemble",
                "task_name": task,
                "dataset": str(dataset.path),
                "query_file": str(query_file),
                "selected_model": best_name,
                "task_type": task_type,
                "target_column": target_col,
                "predictions": [str(x) for x in predictions],
            }
            rel_dir = dataset.path.relative_to(Path(data_root) / task)
            out_path = output / "predict_only_ensemble" / task / rel_dir / f"{query_file.stem}_predict_only.json"
            save_json(result, out_path)
            count += 1
            if max_files is not None and count >= max_files:
                return count
    return count
