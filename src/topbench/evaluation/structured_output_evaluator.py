from __future__ import annotations

from pathlib import Path

import pandas as pd

from topbench.evaluation.metric_utils import f1_from_sets


def evaluate_id_set_csv(prediction_csv: str | Path, target_csv: str | Path, *, id_column: str) -> dict[str, float]:
    pred = pd.read_csv(prediction_csv)
    target = pd.read_csv(target_csv)
    return f1_from_sets(pred[id_column].astype(str), target[id_column].astype(str))
