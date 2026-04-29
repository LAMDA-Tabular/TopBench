from __future__ import annotations

import math
import re
from typing import Any, Iterable, Sequence

import numpy as np


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def classification_accuracy(prediction: Any, target: Any) -> float:
    return 1.0 if normalize_text(prediction) == normalize_text(target) else 0.0


def regression_score(
    prediction: Any,
    target: Any,
    *,
    target_std: float | None = None,
    tolerance: float = 0.05,
) -> float:
    try:
        pred = float(str(prediction).replace(",", ""))
        gt = float(str(target).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0
    if math.isclose(pred, gt, rel_tol=tolerance, abs_tol=1e-8):
        return 1.0
    scale = max(abs(gt), float(target_std or 0.0), 1.0)
    return float(max(0.0, 1.0 - abs(pred - gt) / scale))


def ndcg_at_k(relevances: Sequence[float], k: int) -> float:
    rel = np.asarray(list(relevances)[:k], dtype=float)
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    dcg = float(np.sum(rel * discounts))
    ideal = np.sort(rel)[::-1]
    idcg = float(np.sum(ideal * discounts))
    return dcg / idcg if idcg > 0 else 0.0


def f1_from_sets(predicted: Iterable[Any], target: Iterable[Any]) -> dict[str, float]:
    pred_set = {str(x) for x in predicted}
    gt_set = {str(x) for x in target}
    if not pred_set and not gt_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gt_set) if gt_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
