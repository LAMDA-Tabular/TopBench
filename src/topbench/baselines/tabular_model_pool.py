from __future__ import annotations

from typing import Dict

from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from topbench.baselines.preprocessing import build_preprocessor


def _optional_model_factories(task_type: str) -> Dict[str, object]:
    factories: Dict[str, object] = {}
    try:
        if task_type == "classification":
            from xgboost import XGBClassifier

            factories["xgboost"] = XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss")
        else:
            from xgboost import XGBRegressor

            factories["xgboost"] = XGBRegressor(n_estimators=200, random_state=42)
    except Exception:
        pass
    try:
        if task_type == "classification":
            from lightgbm import LGBMClassifier

            factories["lightgbm"] = LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
        else:
            from lightgbm import LGBMRegressor

            factories["lightgbm"] = LGBMRegressor(n_estimators=200, random_state=42, verbose=-1)
    except Exception:
        pass
    try:
        if task_type == "classification":
            from catboost import CatBoostClassifier

            factories["catboost"] = CatBoostClassifier(iterations=200, random_seed=42, verbose=False)
        else:
            from catboost import CatBoostRegressor

            factories["catboost"] = CatBoostRegressor(iterations=200, random_seed=42, verbose=False)
    except Exception:
        pass
    return factories


def build_model_pool(feature_frame, task_type: str, *, include_optional_models: bool = True) -> Dict[str, Pipeline]:
    preprocessor = build_preprocessor(feature_frame)
    if task_type == "classification":
        estimators = {
            "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=42),
            "extra_trees": ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        }
    else:
        estimators = {
            "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=42),
            "extra_trees": ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        }
    if include_optional_models:
        estimators.update(_optional_model_factories(task_type))
    return {
        name: Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
        for name, estimator in estimators.items()
    }
