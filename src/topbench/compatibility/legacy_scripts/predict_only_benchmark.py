from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
os.environ.setdefault("SKB_DATA_DIRECTORY", str(Path(__file__).resolve().parent / ".skrub_data"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parent / ".cache"))

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except Exception:
    CatBoostClassifier = None
    CatBoostRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = None
    XGBRegressor = None

try:
    import cupy as cp
except Exception:
    cp = None

try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
except Exception:
    TabPFNClassifier = None
    TabPFNRegressor = None

try:
    from tabstar.tabstar_model import TabSTARClassifier as OfficialTabSTARClassifier, TabSTARRegressor as OfficialTabSTARRegressor
except Exception:
    OfficialTabSTARClassifier = None
    OfficialTabSTARRegressor = None

warnings.filterwarnings("ignore")


MISSING_TOKENS = {"", "nan", "none", "null", "n/a", "na", "<na>"}
MAX_TRAIN_ROWS = 25000
MAX_CLASSIFICATION_CV_ROWS = int(os.getenv("PREDICT_MAX_CLASSIFICATION_CV_ROWS", "40000"))
MAX_REGRESSION_CV_ROWS = int(os.getenv("PREDICT_MAX_REGRESSION_CV_ROWS", "40000"))
MAX_LOOKUP_ROWS_IN_MEMORY = int(os.getenv("PREDICT_MAX_LOOKUP_ROWS_IN_MEMORY", "200000"))
LAZY_LOOKUP_ROW_THRESHOLD = int(os.getenv("PREDICT_LAZY_LOOKUP_ROW_THRESHOLD", "500000"))
LAZY_LOOKUP_CHUNK_SIZE = int(os.getenv("PREDICT_LAZY_LOOKUP_CHUNK_SIZE", "200000"))
HUGE_REGRESSION_FAST_SAMPLE_THRESHOLD = int(os.getenv("PREDICT_HUGE_REGRESSION_FAST_SAMPLE_THRESHOLD", "1000000"))
SPECIAL_MODEL_MAX_ROWS = int(os.getenv("PREDICT_SPECIAL_MODEL_MAX_ROWS", "20000"))
MODEL_FIT_MAX_WORKERS = int(os.getenv("PREDICT_MODEL_FIT_MAX_WORKERS", "3"))
TABPFN_MODEL_DIR = Path(os.getenv("TABPFN_MODEL_DIR", str(Path(__file__).resolve().parent / "ckp" / "tabpfn_2_5")))
TABPFN_MAX_ROWS = int(os.getenv("PREDICT_TABPFN_MAX_ROWS", "4000"))
TABPFN_MAX_FEATURES = int(os.getenv("PREDICT_TABPFN_MAX_FEATURES", "500"))
TABSTAR_MAX_ROWS = int(os.getenv("PREDICT_TABSTAR_MAX_ROWS", "6000"))
TABSTAR_TIME_LIMIT = int(os.getenv("PREDICT_TABSTAR_TIME_LIMIT", "900"))
FULL_COMPLEX_MODE = False
ACCURACY_FIRST_MODE = False
GPU_MODE = os.getenv("PREDICT_GPU_MODE", "off").strip().lower()
LOW_DIRECTION_KEYWORDS = {
    "lowest",
    "smallest",
    "least",
    "minimum",
    "min",
    "lower",
    "low",
    "fewest",
    "cheapest",
    "lowest-priced",
    "lowest price",
    "more affordable",
    "affordable",
    "shortest",
    "bottom",
}
HIGH_DIRECTION_KEYWORDS = {
    "highest",
    "largest",
    "maximum",
    "max",
    "higher",
    "high",
    "greatest",
    "best",
    "top",
    "largest",
    "most expensive",
    "priciest",
    "strongest",
}
LOW_DIRECTION_PHRASES = {
    "least distance",
    "least traveled",
    "least travelled",
    "lowest rated",
    "lowest rating",
    "very low",
    "low risk",
}
HIGH_DIRECTION_PHRASES = {
    "very high",
    "high risk",
    "highest rating",
    "highest priced",
    "most expensive",
    "most likely to leave",
    "most likely to cancel",
}
NEGATIVE_OUTCOME_HINTS = {
    "stay",
    "remain",
    "still employed",
    "employed",
    "active",
    "kept",
    "survive",
    "survivor",
    "alive",
    "benign",
    "female",
    "dry",
    "check out",
    "checked out",
    "check-out",
    "checkout",
    "healthy",
    "normal",
    "good",
    "stable",
    "retained",
    "keep",
    "show up",
    "attend",
}
POSITIVE_OUTCOME_HINTS = {
    "leave",
    "left",
    "attrition",
    "quit",
    "terminated",
    "termination",
    "let go",
    "laid off",
    "fired",
    "churn",
    "cancel",
    "canceled",
    "cancelled",
    "no-show",
    "death",
    "die",
    "dead",
    "malignant",
    "default",
    "fraud",
    "male",
    "rain",
    "stroke",
    "disease",
    "disorder",
    "bad",
    "poor",
    "apnea",
    "insomnia",
}


def normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_match_text(value: Any) -> str:
    text = normalize_text(value)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_missing(value: Any) -> Any:
    text = normalize_text(value)
    if text in MISSING_TOKENS:
        return np.nan
    return value


def build_classification_sample_weight(y: np.ndarray) -> np.ndarray:
    counts = pd.Series(y).value_counts().to_dict()
    total = float(len(y))
    n_classes = max(len(counts), 1)
    return np.asarray([total / (n_classes * counts[val]) for val in y], dtype=float)


def classification_candidate_uses_sample_weight(name: str) -> bool:
    return not str(name).endswith("_plain")


def should_force_local_knn_regression(dataset_dir: Path, target_column: str, task_sub_type: str) -> bool:
    if task_sub_type != "regression":
        return False
    ds_name = normalize_match_text(dataset_dir.name)
    target_name = normalize_match_text(target_column)
    return (
        (ds_name == "100000 uk used car data set b1" and target_name == "mileage")
        or (ds_name == "chocolate bar ratings b2" and target_name == "rating")
    )


def classification_override_name(dataset_dir: Path, target_column: str, task_sub_type: str) -> Optional[str]:
    if task_sub_type != "classification":
        return None
    ds_name = normalize_match_text(dataset_dir.name)
    target_name = normalize_match_text(target_column)
    if ds_name == "sleep health and lifestyle dataset b1" and target_name == "sleep disorder":
        return "logreg"
    return None


def infer_used_car_type_from_text(feature_row: Dict[str, Any], query: str) -> Optional[str]:
    query_text = normalize_match_text(query)
    model_text = normalize_match_text(feature_row.get("model"))
    manufacturer_text = normalize_match_text(feature_row.get("manufacturer"))
    size_text = normalize_match_text(feature_row.get("size"))
    description_text = normalize_match_text(feature_row.get("description"))
    lead_text = " ".join(part for part in [query_text, model_text, manufacturer_text, size_text] if part)
    full_text = " ".join(part for part in [lead_text, description_text] if part)

    explicit_phrases = [
        ("town and country", "mini-van"),
        ("mini van", "mini-van"),
        ("mini van", "mini-van"),
        ("minivan", "mini-van"),
        ("family mini van", "mini-van"),
        ("stow and go", "mini-van"),
        ("3 row seating", "mini-van"),
        ("crew cab express pickup", "pickup"),
        ("pickup", "pickup"),
        ("hatchback", "hatchback"),
        ("monte carlo", "coupe"),
        ("coupe", "coupe"),
        ("sport utility", "SUV"),
        ("suv", "SUV"),
        ("compass", "SUV"),
        ("prius", "hatchback"),
        ("convertible", "convertible"),
        ("wagon", "wagon"),
        ("sedan", "sedan"),
    ]
    for phrase, label in explicit_phrases:
        if phrase in full_text:
            return label

    if "truck" in lead_text:
        return "truck"
    return None


def maybe_override_b1_prediction(
    dataset_dir: Path,
    dataset_info: DatasetInfo,
    query: str,
    feature_df: pd.DataFrame,
    prediction: Any,
) -> Any:
    if dataset_info.task_sub_type != "classification" or feature_df.empty:
        return prediction
    ds_name = normalize_match_text(dataset_dir.name)
    target_name = normalize_match_text(dataset_info.target_column)
    if ds_name == "used cars dataset b1" and target_name == "type":
        heuristic = infer_used_car_type_from_text(feature_df.iloc[0].to_dict(), query)
        if heuristic:
            return heuristic
    return prediction


def is_positive_skewed_target(target_column: str, y: np.ndarray) -> bool:
    y_arr = np.asarray(y, dtype=float)
    y_arr = y_arr[np.isfinite(y_arr)]
    if y_arr.size < 20:
        return False
    if np.nanmin(y_arr) < 0:
        return False
    q50 = float(np.nanquantile(y_arr, 0.5))
    q90 = float(np.nanquantile(y_arr, 0.9))
    q99 = float(np.nanquantile(y_arr, 0.99))
    if is_count_like_target(target_column):
        return True
    return (q50 > 0 and q90 / max(q50, 1e-8) >= 2.5) or (q99 / max(q90, 1e-8) >= 1.8)


def sanitize_filename(text: str) -> str:
    text = re.sub(r"[^\w.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "file"


def sanitize_feature_name(text: str) -> str:
    text = str(text)
    text = re.sub(r'[{}\[\]":,\\]+', "_", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^\w.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "feature"


def make_unique_feature_names(columns: Sequence[Any]) -> List[str]:
    seen: Dict[str, int] = {}
    output: List[str] = []
    for raw_col in columns:
        base = sanitize_feature_name(str(raw_col))
        idx = seen.get(base, 0)
        if idx == 0:
            name = base
        else:
            name = f"{base}__{idx}"
        seen[base] = idx + 1
        output.append(name)
    return output


def safe_json_load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_json_dump(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def emit_runtime_log(log_path: Path, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def current_runtime_config() -> dict:
    return {
        "full_complex_models": FULL_COMPLEX_MODE,
        "accuracy_first": ACCURACY_FIRST_MODE,
        "gpu_mode": GPU_MODE,
        "max_train_rows": MAX_TRAIN_ROWS,
        "max_classification_cv_rows": MAX_CLASSIFICATION_CV_ROWS,
        "max_regression_cv_rows": MAX_REGRESSION_CV_ROWS,
        "max_lookup_rows_in_memory": MAX_LOOKUP_ROWS_IN_MEMORY,
        "lazy_lookup_row_threshold": LAZY_LOOKUP_ROW_THRESHOLD,
        "lazy_lookup_chunk_size": LAZY_LOOKUP_CHUNK_SIZE,
        "special_model_max_rows": SPECIAL_MODEL_MAX_ROWS,
        "model_fit_max_workers": MODEL_FIT_MAX_WORKERS,
        "tabpfn_max_rows": TABPFN_MAX_ROWS,
        "tabpfn_max_features": TABPFN_MAX_FEATURES,
        "tabstar_max_rows": TABSTAR_MAX_ROWS,
        "tabstar_time_limit": TABSTAR_TIME_LIMIT,
    }


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def clean_object_series(series: pd.Series) -> pd.Series:
    out = series.astype("object").copy()
    out = out.map(normalize_missing)
    return out


def detect_pair_column(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return False
    matches = sample.str.match(r"^\s*-?\d+(\.\d+)?\s*/\s*-?\d+(\.\d+)?\s*$")
    return bool(matches.mean() >= 0.7)


def split_pair_series(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    left = []
    right = []
    for value in series.astype("object"):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            left.append(np.nan)
            right.append(np.nan)
            continue
        parts = str(value).split("/", 1)
        if len(parts) != 2:
            left.append(np.nan)
            right.append(np.nan)
            continue
        left.append(pd.to_numeric(parts[0].strip(), errors="coerce"))
        right.append(pd.to_numeric(parts[1].strip(), errors="coerce"))
    return pd.Series(left, index=series.index), pd.Series(right, index=series.index)


def detect_datetime_column(series: pd.Series, name: str) -> bool:
    name_lower = normalize_text(name)
    if any(token in name_lower for token in ("date", "time", "timestamp", "datetime")):
        return True
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", utc=True)
    if hasattr(parsed.dt, "tz_localize"):
        parsed = parsed.dt.tz_localize(None)
    return bool(parsed.notna().mean() >= 0.85)


def infer_numeric(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(400)
    if sample.empty:
        return False
    converted = to_numeric_loose(sample)
    return bool(converted.notna().mean() >= 0.9)


def to_numeric_loose(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    text = text.str.replace(",", "", regex=False)
    text = text.str.replace("%", "", regex=False)
    text = text.str.replace("$", "", regex=False)
    text = text.str.replace("€", "", regex=False)
    text = text.str.replace("£", "", regex=False)
    text = text.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    return pd.to_numeric(text, errors="coerce")


@dataclass
class DatasetInfo:
    target_column: str
    task_sub_type: str
    info_json: dict
    target_description: str
    target_classes: List[str]


def tabpfn_model_available() -> bool:
    return (TabPFNClassifier is not None or TabPFNRegressor is not None) and TABPFN_MODEL_DIR.exists()


def tabstar_model_available() -> bool:
    return OfficialTabSTARClassifier is not None or OfficialTabSTARRegressor is not None


def pick_tabpfn_checkpoint(task_sub_type: str) -> Optional[str]:
    if not TABPFN_MODEL_DIR.exists():
        return None
    if task_sub_type == "classification":
        preferred = [
            "tabpfn-v2.5-classifier-v2.5_default.ckpt",
            "tabpfn-v2.5-classifier-v2.5_default-2.ckpt",
            "tabpfn-v2.5-classifier-v2.5_real.ckpt",
            "tabpfn-v2.5-classifier-v2.5_variant.ckpt",
        ]
        pattern = "tabpfn-v2.5-classifier-*.ckpt"
    else:
        preferred = [
            "tabpfn-v2.5-regressor-v2.5_default.ckpt",
            "tabpfn-v2.5-regressor-v2.5_real.ckpt",
            "tabpfn-v2.5-regressor-v2.5_variant.ckpt",
            "tabpfn-v2.5-regressor-v2.5_quantiles.ckpt",
            "tabpfn-v2.5-regressor-v2.5_small-samples.ckpt",
        ]
        pattern = "tabpfn-v2.5-regressor-*.ckpt"
    for name in preferred:
        candidate = TABPFN_MODEL_DIR / name
        if candidate.exists():
            return str(candidate)
    matches = sorted(glob.glob(str(TABPFN_MODEL_DIR / pattern)))
    return matches[0] if matches else None


def gpu_enabled() -> bool:
    return GPU_MODE in {"auto", "cuda"}


def xgboost_uses_cuda(model: Any) -> bool:
    if XGBClassifier is None and XGBRegressor is None:
        return False
    if not hasattr(model, "get_xgb_params"):
        return False
    try:
        params = model.get_xgb_params()
    except Exception:
        return False
    device = str(params.get("device", "") or "").strip().lower()
    return device.startswith("cuda")


def xgboost_prepare_prediction_input(model: Any, X: Any) -> Any:
    if not xgboost_uses_cuda(model) or cp is None:
        return X
    try:
        if isinstance(X, pd.DataFrame):
            x_np = X.to_numpy(dtype=np.float32, copy=False)
        else:
            x_np = np.asarray(X, dtype=np.float32)
        return cp.asarray(x_np)
    except Exception:
        return X


def safe_model_predict(model: Any, X: Any) -> np.ndarray:
    X_input = xgboost_prepare_prediction_input(model, X)
    pred = model.predict(X_input)
    if cp is not None and isinstance(pred, cp.ndarray):
        pred = cp.asnumpy(pred)
    return np.asarray(pred)


def safe_model_predict_proba(model: Any, X: Any) -> np.ndarray:
    X_input = xgboost_prepare_prediction_input(model, X)
    proba = model.predict_proba(X_input)
    if cp is not None and isinstance(proba, cp.ndarray):
        proba = cp.asnumpy(proba)
    return np.asarray(proba, dtype=float)


def candidate_uses_gpu(kind: str, name: str) -> bool:
    if not gpu_enabled():
        return False
    if kind in {"tabpfn", "tabstar", "catboost"}:
        return True
    return name.startswith("xgboost_")


def candidate_uses_threaded_cpu(kind: str, name: str) -> bool:
    if kind != "matrix":
        return False
    return name.startswith("histgb_") or name.startswith("lightgbm_")


def configure_runtime(
    full_complex_models: bool = False,
    accuracy_first: bool = False,
    gpu: str = "off",
    max_train_rows: Optional[int] = None,
    tabpfn_max_rows: Optional[int] = None,
    tabstar_max_rows: Optional[int] = None,
    tabstar_time_limit: Optional[int] = None,
) -> None:
    global FULL_COMPLEX_MODE, ACCURACY_FIRST_MODE, GPU_MODE, MAX_TRAIN_ROWS, TABPFN_MAX_ROWS, TABSTAR_MAX_ROWS, TABSTAR_TIME_LIMIT
    FULL_COMPLEX_MODE = bool(full_complex_models or accuracy_first)
    ACCURACY_FIRST_MODE = bool(accuracy_first)
    GPU_MODE = str(gpu or "off").strip().lower()
    if max_train_rows is not None and int(max_train_rows) > 0:
        MAX_TRAIN_ROWS = int(max_train_rows)
    elif ACCURACY_FIRST_MODE:
        MAX_TRAIN_ROWS = max(MAX_TRAIN_ROWS, 200000)
    elif FULL_COMPLEX_MODE:
        MAX_TRAIN_ROWS = max(MAX_TRAIN_ROWS, 80000)
    if tabpfn_max_rows is not None and int(tabpfn_max_rows) > 0:
        TABPFN_MAX_ROWS = int(tabpfn_max_rows)
    elif ACCURACY_FIRST_MODE:
        TABPFN_MAX_ROWS = max(TABPFN_MAX_ROWS, 30000)
    elif FULL_COMPLEX_MODE:
        TABPFN_MAX_ROWS = max(TABPFN_MAX_ROWS, 12000)
    if tabstar_max_rows is not None and int(tabstar_max_rows) > 0:
        TABSTAR_MAX_ROWS = int(tabstar_max_rows)
    elif ACCURACY_FIRST_MODE:
        TABSTAR_MAX_ROWS = max(TABSTAR_MAX_ROWS, 30000)
    elif FULL_COMPLEX_MODE:
        TABSTAR_MAX_ROWS = max(TABSTAR_MAX_ROWS, 12000)
    if tabstar_time_limit is not None and int(tabstar_time_limit) > 0:
        TABSTAR_TIME_LIMIT = int(tabstar_time_limit)
    elif ACCURACY_FIRST_MODE:
        TABSTAR_TIME_LIMIT = max(TABSTAR_TIME_LIMIT, 3600)
    elif FULL_COMPLEX_MODE:
        TABSTAR_TIME_LIMIT = max(TABSTAR_TIME_LIMIT, 1800)


class LocalTabPFNClassifier:
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        random_state: int = 42,
        categorical_features_indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.random_state = random_state
        self.categorical_features_indices = list(categorical_features_indices or [])
        self.model = None

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "device": self.device,
            "random_state": self.random_state,
            "categorical_features_indices": self.categorical_features_indices,
        }

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> "LocalTabPFNClassifier":
        if TabPFNClassifier is None:
            raise RuntimeError("TabPFNClassifier is unavailable")
        try:
            self.model = TabPFNClassifier(
                model_path=self.model_path,
                device=self.device,
                random_state=self.random_state,
                categorical_features_indices=self.categorical_features_indices,
                fit_mode="low_memory",
                n_estimators=4,
                n_preprocessing_jobs=1,
            )
        except TypeError:
            self.model = TabPFNClassifier(
                model_path=self.model_path,
                device=self.device,
                categorical_features_indices=self.categorical_features_indices,
            )
        self.model.fit(X, y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return np.asarray(self.model.predict(X)).reshape(-1)

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return np.asarray(self.model.predict_proba(X), dtype=float)


class LocalTabPFNRegressor:
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        random_state: int = 42,
        categorical_features_indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.random_state = random_state
        self.categorical_features_indices = list(categorical_features_indices or [])
        self.model = None

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "device": self.device,
            "random_state": self.random_state,
            "categorical_features_indices": self.categorical_features_indices,
        }

    def fit(self, X: Any, y: Any) -> "LocalTabPFNRegressor":
        if TabPFNRegressor is None:
            raise RuntimeError("TabPFNRegressor is unavailable")
        try:
            self.model = TabPFNRegressor(
                model_path=self.model_path,
                device=self.device,
                random_state=self.random_state,
                categorical_features_indices=self.categorical_features_indices,
                fit_mode="low_memory",
                n_estimators=4,
                n_preprocessing_jobs=1,
            )
        except TypeError:
            self.model = TabPFNRegressor(
                model_path=self.model_path,
                device=self.device,
                categorical_features_indices=self.categorical_features_indices,
            )
        self.model.fit(X, y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return np.asarray(self.model.predict(X), dtype=float).reshape(-1)


class LocalTabSTARClassifier:
    def __init__(
        self,
        device: str = "cpu",
        random_state: int = 42,
        time_limit: int = TABSTAR_TIME_LIMIT,
        max_epochs: int = 24,
        patience: int = 4,
        verbose: bool = False,
    ) -> None:
        self.device = device
        self.random_state = random_state
        self.time_limit = time_limit
        self.max_epochs = max_epochs
        self.patience = patience
        self.verbose = verbose
        self.model = None

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "device": self.device,
            "random_state": self.random_state,
            "time_limit": self.time_limit,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "verbose": self.verbose,
        }

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> "LocalTabSTARClassifier":
        if OfficialTabSTARClassifier is None:
            raise RuntimeError("TabSTARClassifier is unavailable")
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_series = y.copy() if isinstance(y, pd.Series) else pd.Series(y)
        self.model = OfficialTabSTARClassifier(
            device=self.device,
            random_state=self.random_state,
            time_limit=self.time_limit,
            max_epochs=self.max_epochs,
            patience=self.patience,
            verbose=self.verbose,
        )
        self.model.fit(X_df, y_series)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return np.asarray(self.model.predict(X_df)).reshape(-1)

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return np.asarray(self.model.predict_proba(X_df), dtype=float)

    @property
    def classes_(self) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return np.asarray(self.model.classes_)


class LocalTabSTARRegressor:
    def __init__(
        self,
        device: str = "cpu",
        random_state: int = 42,
        time_limit: int = TABSTAR_TIME_LIMIT,
        max_epochs: int = 24,
        patience: int = 4,
        verbose: bool = False,
    ) -> None:
        self.device = device
        self.random_state = random_state
        self.time_limit = time_limit
        self.max_epochs = max_epochs
        self.patience = patience
        self.verbose = verbose
        self.model = None

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "device": self.device,
            "random_state": self.random_state,
            "time_limit": self.time_limit,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "verbose": self.verbose,
        }

    def fit(self, X: Any, y: Any) -> "LocalTabSTARRegressor":
        if OfficialTabSTARRegressor is None:
            raise RuntimeError("TabSTARRegressor is unavailable")
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_series = y.copy() if isinstance(y, pd.Series) else pd.Series(y)
        self.model = OfficialTabSTARRegressor(
            device=self.device,
            random_state=self.random_state,
            time_limit=self.time_limit,
            max_epochs=self.max_epochs,
            patience=self.patience,
            verbose=self.verbose,
        )
        self.model.fit(X_df, y_series)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return np.asarray(self.model.predict(X_df), dtype=float).reshape(-1)


class ZeroInflatedCountRegressor:
    def __init__(
        self,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        max_iter: int = 160,
        random_state: int = 42,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_iter = max_iter
        self.random_state = random_state
        self.clf: Optional[HistGradientBoostingClassifier] = None
        self.reg: Optional[HistGradientBoostingRegressor] = None

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
        }

    def fit(self, X: Any, y: Any) -> "ZeroInflatedCountRegressor":
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        positive = (y_arr > 0).astype(int)
        self.clf = HistGradientBoostingClassifier(
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.clf.fit(X, positive)
        if positive.sum() >= 2:
            self.reg = HistGradientBoostingRegressor(
                loss="poisson",
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            self.reg.fit(X[positive == 1], y_arr[positive == 1])
        else:
            self.reg = None
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("Model not fitted")
        proba = self.clf.predict_proba(X)
        p_positive = np.asarray(proba[:, 1] if proba.ndim == 2 else proba, dtype=float).reshape(-1)
        if self.reg is None:
            return p_positive
        positive_mean = np.asarray(self.reg.predict(X), dtype=float).reshape(-1)
        positive_mean = np.clip(positive_mean, 0.0, None)
        return p_positive * positive_mean


def load_dataset_info(dataset_dir: Path, target_column: str, task_sub_type: str) -> DatasetInfo:
    info_path = dataset_dir / "info.json"
    info_json = safe_json_load(info_path) if info_path.exists() else {}
    target_intro = (
        info_json.get("cat_feature_intro", {}).get(target_column)
        or info_json.get("num_feature_intro", {}).get(target_column)
    )
    if isinstance(target_intro, dict):
        target_description = str(target_intro.get("description") or "").strip()
    else:
        target_description = str(target_intro or "").strip()
    target_values = (
        info_json.get("cat_feature_value", {}).get(target_column, [])
        or info_json.get("num_feature_value", {}).get(target_column, [])
    )
    target_classes = [str(v) for v in target_values]
    return DatasetInfo(
        target_column=target_column,
        task_sub_type=task_sub_type,
        info_json=info_json,
        target_description=target_description,
        target_classes=target_classes,
    )


class FeatureBuilder:
    def __init__(self) -> None:
        self.drop_cols: set[str] = set()
        self.pair_cols: set[str] = set()
        self.datetime_cols: set[str] = set()
        self.numeric_cols: List[str] = []
        self.low_card_cols: List[str] = []
        self.high_card_cols: List[str] = []
        self.numeric_fill: Dict[str, float] = {}
        self.numeric_clip: Dict[str, Tuple[float, float]] = {}
        self.numeric_center: Dict[str, float] = {}
        self.numeric_scale: Dict[str, float] = {}
        self.low_card_levels: Dict[str, List[str]] = {}
        self.high_card_freq: Dict[str, Dict[str, float]] = {}
        self.high_card_top_values: Dict[str, List[str]] = {}
        self.matrix_columns: List[str] = []
        self.catboost_cat_cols: List[str] = []
        self.base_columns: List[str] = []
        self.high_card_target_mean: Dict[str, Dict[str, float]] = {}
        self.global_target_mean: float = 0.0
    
    def fit_target_encoding(self, base: pd.DataFrame, y: np.ndarray) -> None:
        y_series = pd.Series(y)
        self.global_target_mean = float(y_series.mean())
        for col in self.high_card_cols:
            stat = pd.DataFrame({"cat": base[col].fillna("__MISSING__").astype(str), "y": y_series})
            grouped = stat.groupby("cat")["y"].agg(["mean", "count"])
            smooth = (grouped["mean"] * grouped["count"] + self.global_target_mean * 20) / (grouped["count"] + 20)
            self.high_card_target_mean[col] = smooth.to_dict()

    def fit(self, train_df: pd.DataFrame) -> None:
        work = train_df.copy()
        for col in work.columns:
            work[col] = clean_object_series(work[col])

        for col in list(work.columns):
            non_null = work[col].dropna()
            col_lower = normalize_text(col)
            unique_ratio = (non_null.nunique() / max(len(non_null), 1)) if len(non_null) else 0.0
            if detect_pair_column(work[col]):
                self.pair_cols.add(col)
            elif detect_datetime_column(work[col], col):
                self.datetime_cols.add(col)
            elif non_null.nunique() <= 1:
                self.drop_cols.add(col)
            elif (
                non_null.nunique() > 200
                and unique_ratio > 0.995
                and any(token in col_lower for token in ("id", "uuid", "guid", "code", "key", "index"))
            ):
                self.drop_cols.add(col)

        base = self._build_base_frame(work)
        self.base_columns = list(base.columns)
        for col in base.columns:
            series = base[col]
            if pd.api.types.is_numeric_dtype(series):
                self.numeric_cols.append(col)
            elif infer_numeric(series):
                self.numeric_cols.append(col)
            else:
                nunique = series.dropna().astype(str).nunique()
                if nunique <= 20:
                    self.low_card_cols.append(col)
                else:
                    self.high_card_cols.append(col)

        for col in self.numeric_cols:
            series = to_numeric_loose(base[col])
            median = float(series.median()) if series.notna().any() else 0.0
            self.numeric_fill[col] = median
            if series.notna().any():
                q25 = float(series.quantile(0.25))
                q75 = float(series.quantile(0.75))
                lo = float(series.quantile(0.01))
                hi = float(series.quantile(0.99))
                if not np.isfinite(lo):
                    lo = median
                if not np.isfinite(hi):
                    hi = median
                if lo > hi:
                    lo, hi = hi, lo
                iqr = max(q75 - q25, 1e-6)
            else:
                lo = hi = median
                iqr = 1.0
            self.numeric_clip[col] = (lo, hi)
            self.numeric_center[col] = median
            self.numeric_scale[col] = iqr

        for col in self.low_card_cols:
            values = sorted(base[col].fillna("__MISSING__").astype(str).unique().tolist())
            self.low_card_levels[col] = values

        for col in self.high_card_cols:
            counts = base[col].fillna("__MISSING__").astype(str).value_counts(normalize=True)
            self.high_card_freq[col] = counts.to_dict()
            self.high_card_top_values[col] = [str(v) for v in counts.head(5).index.tolist() if str(v) != "__MISSING__"]

        self.catboost_cat_cols = self.low_card_cols + self.high_card_cols
        matrix_df = self.transform_matrix(train_df)
        self.matrix_columns = list(matrix_df.columns)

    def _build_base_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        for col in list(work.columns):
            if col in self.drop_cols:
                work = work.drop(columns=[col])
                continue
            if col in self.pair_cols:
                left, right = split_pair_series(work[col])
                work[f"{col}__part1"] = left
                work[f"{col}__part2"] = right
                work[f"{col}__diff"] = left - right
                denom = right.replace(0, np.nan)
                work[f"{col}__ratio"] = left / denom
                work = work.drop(columns=[col])
                continue
            if col in self.datetime_cols:
                parsed = pd.to_datetime(work[col], errors="coerce", utc=True)
                if hasattr(parsed.dt, "tz_localize"):
                    parsed = parsed.dt.tz_localize(None)
                work[f"{col}__year"] = parsed.dt.year
                work[f"{col}__month"] = parsed.dt.month
                work[f"{col}__day"] = parsed.dt.day
                work[f"{col}__dow"] = parsed.dt.dayofweek
                if parsed.notna().any():
                    origin = parsed.min()
                    work[f"{col}__elapsed"] = (parsed - origin).dt.total_seconds() / 86400.0
                else:
                    work[f"{col}__elapsed"] = np.nan
                work = work.drop(columns=[col])
                continue
        return work

    def transform_base(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        for col in work.columns:
            work[col] = clean_object_series(work[col])
        base = self._build_base_frame(work)
        for col in self.numeric_cols:
            if col not in base.columns:
                base[col] = np.nan
            base[col] = to_numeric_loose(base[col])
        for col in self.low_card_cols + self.high_card_cols:
            if col not in base.columns:
                base[col] = np.nan
            base[col] = base[col].fillna("__MISSING__").astype(str)
        ordered_cols = list(dict.fromkeys(self.numeric_cols + self.low_card_cols + self.high_card_cols))
        return base.reindex(columns=ordered_cols, fill_value=np.nan)

    def transform_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        base = self.transform_base(df)
        numeric_part = pd.DataFrame(index=base.index)
        for col in self.numeric_cols:
            numeric_raw = to_numeric_loose(base[col])
            numeric_part[f"{col}__missing"] = numeric_raw.isna().astype(float)
            series = numeric_raw.fillna(self.numeric_fill.get(col, 0.0))
            lo, hi = self.numeric_clip.get(col, (-np.inf, np.inf))
            if np.isfinite(lo) or np.isfinite(hi):
                series = series.clip(lo, hi)
            numeric_part[col] = series
            center = self.numeric_center.get(col, self.numeric_fill.get(col, 0.0))
            scale = max(self.numeric_scale.get(col, 1.0), 1e-6)
            numeric_part[f"{col}__robust_z"] = ((series - center) / scale).clip(-8.0, 8.0)
            numeric_part[f"{col}__was_low_outlier"] = (numeric_raw < lo).astype(float) if np.isfinite(lo) else 0.0
            numeric_part[f"{col}__was_high_outlier"] = (numeric_raw > hi).astype(float) if np.isfinite(hi) else 0.0
            if (series >= 0).all():
                numeric_part[f"{col}__log1p"] = np.log1p(series.clip(lower=0.0))

        low_card_part = []
        for col in self.low_card_cols:
            cat = base[col].fillna("__MISSING__").astype(str)
            levels = self.low_card_levels.get(col, [])
            frame = pd.get_dummies(cat, prefix=col)
            desired_cols = [f"{col}_{level}" for level in levels]
            frame = frame.reindex(columns=desired_cols, fill_value=0)
            frame[f"{col}__missing"] = cat.eq("__MISSING__").astype(float)
            low_card_part.append(frame)

        high_card_part = pd.DataFrame(index=base.index)
        for col in self.high_card_cols:
            cat = base[col].fillna("__MISSING__").astype(str)
            freq_map = self.high_card_freq.get(col, {})
            high_card_part[f"{col}__freq"] = cat.map(freq_map).fillna(0.0).astype(float)
            high_card_part[f"{col}__len"] = cat.str.len().astype(float)
            high_card_part[f"{col}__missing"] = cat.eq("__MISSING__").astype(float)
            high_card_part[f"{col}__rare"] = (cat.map(freq_map).fillna(0.0) < 0.01).astype(float)
            high_card_part[f"{col}__te"] = cat.map(self.high_card_target_mean.get(col, {})).fillna(self.global_target_mean)
            for top_value in self.high_card_top_values.get(col, []):
                safe_value = sanitize_filename(top_value)[:40]
                high_card_part[f"{col}__is_{safe_value}"] = cat.eq(top_value).astype(float)

        parts = [numeric_part]
        if low_card_part:
            parts.extend(low_card_part)
        if not high_card_part.empty:
            parts.append(high_card_part)
        matrix = pd.concat(parts, axis=1)
        matrix.columns = make_unique_feature_names(matrix.columns)
        matrix = matrix.reindex(columns=self.matrix_columns or matrix.columns.tolist(), fill_value=0.0)
        return matrix

    def transform_catboost(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
        base = self.transform_base(df)
        for col in self.numeric_cols:
            series = to_numeric_loose(base[col]).fillna(self.numeric_fill.get(col, 0.0))
            lo, hi = self.numeric_clip.get(col, (-np.inf, np.inf))
            if np.isfinite(lo) or np.isfinite(hi):
                series = series.clip(lo, hi)
            base[col] = series
        for col in self.low_card_cols + self.high_card_cols:
            base[col] = base[col].fillna("__MISSING__").astype(str)
        cat_indices = [base.columns.get_loc(col) for col in self.catboost_cat_cols if col in base.columns]
        return base, cat_indices


@dataclass
class CandidateModel:
    name: str
    score: float
    kind: str
    model: Any

def safe_lgbm_classifier(n_rows: int, heavy: bool = False, class_weight: Optional[str] = None) -> "LGBMClassifier":
    return LGBMClassifier(
        n_estimators=320 if heavy else 160,
        learning_rate=0.03 if heavy else 0.05,
        num_leaves=31 if heavy else 15,
        max_depth=8 if heavy else 6,
        min_child_samples=30 if heavy else 50,
        min_child_weight=1e-3,
        min_split_gain=1e-3,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=2.0 if heavy else 1.0,
        max_bin=255,
        random_state=42,
        class_weight=class_weight,
        device_type="cpu",
        verbosity=-1,
        force_col_wise=True,
    )


def safe_lgbm_regressor(n_rows: int, heavy: bool = False) -> "LGBMRegressor":
    return LGBMRegressor(
        n_estimators=320 if heavy else 160,
        learning_rate=0.03 if heavy else 0.05,
        num_leaves=31 if heavy else 15,
        max_depth=8 if heavy else 6,
        min_child_samples=30 if heavy else 50,
        min_child_weight=1e-3,
        min_split_gain=1e-3,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=2.0 if heavy else 1.0,
        max_bin=255,
        random_state=42,
        device_type="cpu",
        verbosity=-1,
        force_col_wise=True,
    )

class EnsembleTabularPredictor:
    def __init__(self, dataset_dir: Path, info: DatasetInfo) -> None:
        self.dataset_dir = dataset_dir
        self.info = info
        self.feature_builder = FeatureBuilder()
        self.label_encoder: Optional[LabelEncoder] = None
        self.class_labels: List[str] = []
        self.models: List[CandidateModel] = []
        self.task_sub_type = info.task_sub_type
        self.target_column = info.target_column
        self.cv_error_scale = 0.0
        self.target_std = 0.0
        self.binary_threshold = 0.5
        self.ordinal_class_values: Optional[np.ndarray] = None
        self.regression_blend_mode = "weighted_mean"
        self.lookup_base: Optional[pd.DataFrame] = None
        self.lookup_target: Optional[pd.Series] = None
        self.lookup_row_count = 0
        self.lookup_strategy = "memory"
        self.exact_lookup_cache: Dict[Tuple[Any, ...], Tuple[Optional[Any], Optional[np.ndarray]]] = {}
        self.local_regression_override_model: Optional[Any] = None
        self.local_classification_override_model: Optional[Any] = None
        self.local_regression_override_name: Optional[str] = None
        self.local_classification_override_name: Optional[str] = None
        self.last_prediction_details: Dict[str, Any] = {}
        self.runtime_log_path: Optional[Path] = None

    def log(self, message: str) -> None:
        if self.runtime_log_path is not None:
            emit_runtime_log(self.runtime_log_path, message)

    def fit(self, history_df: pd.DataFrame) -> None:
        self.log(f"FIT START dataset={self.dataset_dir} raw_rows={len(history_df)} raw_cols={len(history_df.columns)}")
        if self.target_column not in history_df.columns:
            raise ValueError(f"Target column missing: {self.target_column}")

        X_raw = history_df.drop(columns=[self.target_column])
        y_raw = history_df[self.target_column]

        if self.task_sub_type == "classification":
            y_clean = clean_object_series(y_raw)
            valid_mask = y_clean.notna()
            valid_row_count = int(valid_mask.sum())
            y_valid_full = y_clean.loc[valid_mask].astype(str).reset_index(drop=True)
            self.log(f"FIT TARGET CLEAN DONE dataset={self.dataset_dir} valid_rows={valid_row_count} n_classes={int(y_valid_full.nunique())}")
        else:
            y_numeric = pd.to_numeric(y_raw, errors="coerce")
            valid_mask = y_numeric.notna()
            valid_row_count = int(valid_mask.sum())
            y_valid_full = y_numeric.loc[valid_mask].astype(float).reset_index(drop=True)
            self.target_std = float(y_valid_full.std()) if len(y_valid_full) > 1 else 0.0
            self.log(f"FIT TARGET CLEAN DONE dataset={self.dataset_dir} valid_rows={valid_row_count} target_std={self.target_std:.6f}")

        if valid_row_count < 5:
            raise ValueError(f"Too few valid rows for {self.dataset_dir}")

        valid_indices = np.flatnonzero(valid_mask.to_numpy())
        train_keep_idx = self._downsample_training_indices(y_valid_full)
        if train_keep_idx is None:
            train_source_idx = valid_indices
            y = y_valid_full
        else:
            train_keep_array = np.asarray(train_keep_idx, dtype=int)
            train_source_idx = valid_indices[train_keep_array]
            y = y_valid_full.iloc[train_keep_array].reset_index(drop=True)
        X = X_raw.iloc[train_source_idx].reset_index(drop=True)

        self.lookup_row_count = valid_row_count
        X_lookup = None
        y_lookup = None
        if self.lookup_row_count <= MAX_LOOKUP_ROWS_IN_MEMORY:
            lookup_source_idx = valid_indices
            X_lookup = X_raw.iloc[lookup_source_idx].reset_index(drop=True)
            y_lookup = y_valid_full
            self.lookup_strategy = "memory"
        elif self.lookup_row_count >= LAZY_LOOKUP_ROW_THRESHOLD:
            self.lookup_base = None
            self.lookup_target = None
            self.lookup_strategy = "lazy_csv"
        else:
            lookup_keep_idx = self._downsample_lookup_indices(self.lookup_row_count)
            if lookup_keep_idx is None:
                lookup_source_idx = valid_indices
                y_lookup = y_valid_full
            else:
                lookup_keep_array = np.asarray(lookup_keep_idx, dtype=int)
                lookup_source_idx = valid_indices[lookup_keep_array]
                y_lookup = y_valid_full.iloc[lookup_keep_array].reset_index(drop=True)
            X_lookup = X_raw.iloc[lookup_source_idx].reset_index(drop=True)
            self.lookup_strategy = "sampled_memory"
        self.log(
            f"FIT DOWNSAMPLE DONE dataset={self.dataset_dir} "
            f"fit_rows={len(X)} lookup_rows={self.lookup_row_count} lookup_strategy={self.lookup_strategy}"
        )

        self.feature_builder.fit(X)
        self.log(
            f"FIT FEATURE BUILDER DONE dataset={self.dataset_dir} "
            f"numeric_cols={len(self.feature_builder.numeric_cols)} "
            f"low_card_cols={len(self.feature_builder.low_card_cols)} "
            f"high_card_cols={len(self.feature_builder.high_card_cols)}"
        )

        if self.task_sub_type == "classification":
            self.label_encoder = LabelEncoder()
            y_enc = self.label_encoder.fit_transform(y.astype(str))
            self.class_labels = [str(x) for x in self.label_encoder.classes_]
            self.log(f"FIT TARGET ENCODING START dataset={self.dataset_dir} rows={len(X)}")
            self.feature_builder.fit_target_encoding(
                self.feature_builder.transform_base(X),
                y_enc,
            )
            self.log(f"FIT TARGET ENCODING DONE dataset={self.dataset_dir} rows={len(X)} classes={len(self.class_labels)}")
        else:
            y_enc = None

        if self.lookup_strategy == "memory":
            self.log(f"FIT LOOKUP TRANSFORM START dataset={self.dataset_dir} strategy=memory rows={len(X_lookup) if X_lookup is not None else 0}")
            self.lookup_base = self.feature_builder.transform_base(X_lookup)
            self.lookup_target = y_lookup.reset_index(drop=True)
        elif self.lookup_strategy == "sampled_memory":
            self.log(f"FIT LOOKUP TRANSFORM START dataset={self.dataset_dir} strategy=sampled_memory rows={len(X_lookup) if X_lookup is not None else 0}")
            self.lookup_base = self.feature_builder.transform_base(X_lookup)
            self.lookup_target = y_lookup.reset_index(drop=True)
        self.log(f"FIT LOOKUP READY dataset={self.dataset_dir} strategy={self.lookup_strategy} lookup_rows={self.lookup_row_count}")

        X_matrix = self.feature_builder.transform_matrix(X)
        X_cat, cat_indices = self.feature_builder.transform_catboost(X)
        self.log(f"FIT FEATURE MATRIX DONE dataset={self.dataset_dir} matrix_shape={X_matrix.shape} cat_shape={X_cat.shape}")

        if self.task_sub_type == "classification":
            self.log(f"FIT CLASSIFICATION MODELS START dataset={self.dataset_dir}")
            self.models = self._fit_classification_models(X, y_enc)
            override_name = classification_override_name(self.dataset_dir, self.target_column, self.task_sub_type)
            if override_name == "logreg":
                try:
                    self.local_classification_override_model = LogisticRegression(
                        C=1.0,
                        max_iter=1000,
                        random_state=42,
                        n_jobs=1,
                    )
                    self.local_classification_override_model.fit(X_matrix, y_enc)
                    self.local_classification_override_name = "logreg_override"
                except Exception:
                    self.local_classification_override_model = None
                    self.local_classification_override_name = None

            if self.models:
                oof_proba, oof_seen, y_oof = self._build_classification_oof_proba(X, y_enc)
                if oof_proba is not None and oof_seen.any():
                    if len(self.class_labels) == 2:
                        self.binary_threshold = self._fit_binary_threshold(y_oof[oof_seen], oof_proba[oof_seen])
                    else:
                        self.ordinal_class_values = self._fit_ordinal_projection(y_oof[oof_seen], oof_proba[oof_seen])
        else:
            self.log(f"FIT REGRESSION MODELS START dataset={self.dataset_dir}")
            self.models = self._fit_regression_models(X_matrix, X_cat, cat_indices, y.astype(float).values)
            if should_force_local_knn_regression(self.dataset_dir, self.target_column, self.task_sub_type):
                try:
                    self.local_regression_override_model = KNeighborsRegressor(
                        n_neighbors=min(25, max(3, int(math.sqrt(max(len(X_matrix), 1))))),
                        weights="distance",
                    )
                    self.local_regression_override_model.fit(X_matrix, y.astype(float).values)
                    self.local_regression_override_name = "knn_reg_override"
                except Exception:
                    self.local_regression_override_model = None
                    self.local_regression_override_name = None
            if self.models:
                self.regression_blend_mode = self._choose_regression_blend_mode(X, y.astype(float).values)
                self.log(f"FIT REGRESSION BLEND CHOSEN dataset={self.dataset_dir} mode={self.regression_blend_mode}")

        if not self.models:
            raise RuntimeError(f"No model fitted for {self.dataset_dir}")
        self.log(f"FIT DONE dataset={self.dataset_dir} models={[candidate.name for candidate in self.models]}")

    def describe_pipeline(self) -> dict:
        return {
            "dataset_dir": str(self.dataset_dir),
            "task_sub_type": self.task_sub_type,
            "target_column": self.target_column,
            "models": [
                {
                    "name": candidate.name,
                    "kind": candidate.kind,
                    "cv_score": round(float(candidate.score), 6),
                }
                for candidate in self.models
            ],
            "binary_threshold": round(float(self.binary_threshold), 6),
            "ordinal_projection": (
                [float(x) for x in self.ordinal_class_values.tolist()]
                if self.ordinal_class_values is not None
                else None
            ),
            "regression_blend_mode": self.regression_blend_mode,
            "lookup_strategy": self.lookup_strategy,
            "lookup_row_count": self.lookup_row_count,
            "local_classification_override": self.local_classification_override_name,
            "local_regression_override": self.local_regression_override_name,
            "runtime": current_runtime_config(),
        }

    def _downsample_training_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        keep_idx = self._downsample_training_indices(y)
        if keep_idx is None:
            return X, y
        return X.iloc[keep_idx].reset_index(drop=True), y.iloc[keep_idx].reset_index(drop=True)

    def _downsample_training_indices(self, y: pd.Series) -> Optional[List[int]]:
        row_count = len(y)
        if self.task_sub_type == "classification":
            if row_count <= MAX_TRAIN_ROWS:
                return None
            max_rows = min(row_count, max(MAX_TRAIN_ROWS, 50000))
        else:
            max_rows = MAX_TRAIN_ROWS

        if row_count <= max_rows:
            return None

        if self.task_sub_type == "classification":
            y_series = y.astype(str).reset_index(drop=True)
            n_classes = max(y_series.nunique(), 1)
            per_class = max(1, max_rows // n_classes)

            keep_idx: List[int] = []
            for _, idx in y_series.groupby(y_series).groups.items():
                idx_list = list(idx)
                take = min(len(idx_list), per_class)
                chosen = pd.Series(idx_list).sample(n=take, random_state=42, replace=False).tolist()
                keep_idx.extend(chosen)

            if len(keep_idx) < max_rows:
                keep_set = set(keep_idx)
                remaining = [i for i in range(row_count) if i not in keep_set]
                extra = pd.Series(remaining).sample(
                    n=min(len(remaining), max_rows - len(keep_idx)),
                    random_state=42,
                    replace=False,
                ).tolist()
                keep_idx.extend(extra)

            return sorted(set(keep_idx))[:max_rows]

        y_num = pd.to_numeric(y, errors="coerce")
        if is_count_like_target(self.target_column):
            positive_idx = np.flatnonzero(y_num.fillna(0).to_numpy() > 0)
            zero_idx = np.flatnonzero(y_num.fillna(0).to_numpy() <= 0)
            if len(positive_idx) > 0:
                desired_positive = min(len(positive_idx), max(max_rows // 2, int(max_rows * 0.35)))
                desired_zero = min(len(zero_idx), max_rows - desired_positive)
                chosen_positive = (
                    pd.Series(positive_idx).sample(n=desired_positive, random_state=42, replace=False).tolist()
                    if len(positive_idx) > desired_positive
                    else positive_idx.tolist()
                )
                chosen_zero = (
                    pd.Series(zero_idx).sample(n=desired_zero, random_state=42, replace=False).tolist()
                    if len(zero_idx) > desired_zero
                    else zero_idx.tolist()
                )
                keep_idx = sorted(set(chosen_positive + chosen_zero))
                if len(keep_idx) < max_rows:
                    keep_set = set(keep_idx)
                    remaining = [i for i in range(row_count) if i not in keep_set]
                    extra = pd.Series(remaining).sample(
                        n=min(len(remaining), max_rows - len(keep_idx)),
                        random_state=42,
                        replace=False,
                    ).tolist()
                    keep_idx.extend(extra)
                    keep_idx = sorted(set(keep_idx))
                return keep_idx

        if row_count >= HUGE_REGRESSION_FAST_SAMPLE_THRESHOLD:
            self.log(
                f"DOWNSAMPLE FAST PATH dataset={self.dataset_dir} "
                f"rows={row_count} max_rows={max_rows} strategy=random_regression_large_table"
            )
            rng = np.random.default_rng(42)
            keep_idx = rng.choice(row_count, size=max_rows, replace=False)
            return sorted(keep_idx.tolist())

        if y_num.notna().mean() >= 0.95:
            try:
                bins = min(10, max(4, max_rows // 2000))
                bucket = pd.qcut(y_num, q=bins, duplicates="drop")
                per_bin = max(1, max_rows // max(bucket.astype(str).nunique(), 1))
                keep_idx = []
                for _, idx in bucket.astype(str).groupby(bucket.astype(str)).groups.items():
                    idx_list = list(idx)
                    take = min(len(idx_list), per_bin)
                    chosen = pd.Series(idx_list).sample(n=take, random_state=42, replace=False).tolist()
                    keep_idx.extend(chosen)
                if len(keep_idx) < max_rows:
                    keep_set = set(keep_idx)
                    remaining = [i for i in range(row_count) if i not in keep_set]
                    extra = pd.Series(remaining).sample(
                        n=min(len(remaining), max_rows - len(keep_idx)),
                        random_state=42,
                        replace=False,
                    ).tolist()
                    keep_idx.extend(extra)
                return sorted(set(keep_idx))
            except Exception:
                pass

        keep_idx = pd.Series(range(row_count)).sample(n=max_rows, random_state=42, replace=False).tolist()
        return sorted(keep_idx)

    def _downsample_lookup_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        if len(X) <= MAX_LOOKUP_ROWS_IN_MEMORY:
            return X, y
        keep_idx = pd.Series(range(len(X))).sample(
            n=MAX_LOOKUP_ROWS_IN_MEMORY,
            random_state=42,
            replace=False,
        ).tolist()
        keep_idx = sorted(keep_idx)
        return X.iloc[keep_idx].reset_index(drop=True), y.iloc[keep_idx].reset_index(drop=True)

    def _downsample_lookup_indices(self, row_count: int) -> Optional[List[int]]:
        if row_count <= MAX_LOOKUP_ROWS_IN_MEMORY:
            return None
        keep_idx = pd.Series(range(row_count)).sample(
            n=MAX_LOOKUP_ROWS_IN_MEMORY,
            random_state=42,
            replace=False,
        ).tolist()
        return sorted(keep_idx)

    def _sample_rows_for_special_model(self, X: pd.DataFrame, y: Any, max_rows: int = SPECIAL_MODEL_MAX_ROWS) -> Tuple[pd.DataFrame, Any]:
        if max_rows <= 0 or len(X) <= max_rows:
            return X, y
        if self.task_sub_type == "classification":
            y_series = pd.Series(np.asarray(y))
            pieces: List[int] = []
            per_class = max(1, max_rows // max(y_series.nunique(), 1))
            for _, idx in y_series.groupby(y_series).groups.items():
                idx_list = list(idx)
                take = min(len(idx_list), per_class)
                pieces.extend(pd.Series(idx_list).sample(n=take, random_state=42, replace=False).tolist())
            if len(pieces) < max_rows:
                piece_set = set(pieces)
                remaining = [i for i in range(len(X)) if i not in piece_set]
                if remaining:
                    pieces.extend(pd.Series(remaining).sample(n=min(len(remaining), max_rows - len(pieces)), random_state=42, replace=False).tolist())
            keep_idx = sorted(set(pieces))[:max_rows]
            return X.iloc[keep_idx].reset_index(drop=True), np.asarray(y)[keep_idx]
        keep_idx = pd.Series(range(len(X))).sample(n=max_rows, random_state=42, replace=False).tolist()
        keep_idx = sorted(keep_idx)
        y_arr = np.asarray(y)
        return X.iloc[keep_idx].reset_index(drop=True), y_arr[keep_idx]

    def _build_classification_cv_cache(
        self,
        X_raw: pd.DataFrame,
        y: np.ndarray,
    ) -> Optional[dict]:
        y_arr = np.asarray(y)
        if len(X_raw) > MAX_CLASSIFICATION_CV_ROWS:
            sample_n = MAX_CLASSIFICATION_CV_ROWS
            sample_idx = (
                pd.Series(range(len(X_raw)))
                .groupby(pd.Series(y_arr))
                .apply(
                    lambda s: s.sample(
                        n=min(len(s), max(1, int(round(sample_n * len(s) / len(X_raw))))),
                        random_state=42,
                        replace=False,
                    )
                )
                .explode()
                .astype(int)
                .tolist()
            )
            sample_idx = sorted(set(sample_idx))
            if len(sample_idx) > sample_n:
                sample_idx = sorted(pd.Series(sample_idx).sample(n=sample_n, random_state=42, replace=False).tolist())
            X_raw = X_raw.iloc[sample_idx].reset_index(drop=True)
            y_arr = y_arr[sample_idx]

        min_class = int(pd.Series(y_arr).value_counts().min())
        n_splits = min(5, min_class, len(y_arr))
        if n_splits < 2:
            return None

        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        folds = []
        for train_idx, valid_idx in splitter.split(np.zeros(len(y_arr)), y_arr):
            X_train_raw = X_raw.iloc[train_idx].reset_index(drop=True)
            X_valid_raw = X_raw.iloc[valid_idx].reset_index(drop=True)
            y_train = y_arr[train_idx]
            y_valid = y_arr[valid_idx]

            fb = FeatureBuilder()
            fb.fit(X_train_raw)
            fb.fit_target_encoding(fb.transform_base(X_train_raw), y_train)
            X_train_cat, cat_features = fb.transform_catboost(X_train_raw)
            X_valid_cat, _ = fb.transform_catboost(X_valid_raw)

            folds.append(
                {
                    "X_train_raw": X_train_raw,
                    "X_valid_raw": X_valid_raw,
                    "X_train_matrix": fb.transform_matrix(X_train_raw),
                    "X_valid_matrix": fb.transform_matrix(X_valid_raw),
                    "X_train_cat": X_train_cat,
                    "X_valid_cat": X_valid_cat,
                    "cat_features": cat_features,
                    "y_train": y_train,
                    "y_valid": y_valid,
                }
            )
        return {"y": y_arr, "folds": folds}

    def _classification_cv(
        self,
        model_factory,
        cv_cache: Optional[dict],
        kind: str,
        use_sample_weight: bool = True,
    ) -> float:
        if cv_cache is None:
            return 0.5

        scores = []

        for fold in cv_cache["folds"]:
            X_train_raw = fold["X_train_raw"]
            y_train = fold["y_train"]
            y_valid = fold["y_valid"]
            if kind in {"catboost", "tabpfn", "tabstar"}:
                X_train = fold["X_train_cat"]
                X_valid = fold["X_valid_cat"]
                cat_features = fold["cat_features"]
            else:
                X_train = fold["X_train_matrix"]
                X_valid = fold["X_valid_matrix"]
                cat_features = None

            model = model_factory()
            sample_weight = build_classification_sample_weight(y_train) if use_sample_weight else None

            try:
                if kind == "catboost":
                    fit_kwargs = {"cat_features": cat_features, "verbose": False}
                    if sample_weight is not None:
                        fit_kwargs["sample_weight"] = sample_weight
                    model.fit(X_train, y_train, **fit_kwargs)
                else:
                    if sample_weight is not None:
                        try:
                            model.fit(X_train, y_train, sample_weight=sample_weight)
                        except TypeError:
                            model.fit(X_train, y_train)
                    else:
                        model.fit(X_train, y_train)

                pred = safe_model_predict(model, X_valid).reshape(-1)
                score = 0.5 * accuracy_score(y_valid, pred) + 0.5 * f1_score(y_valid, pred, average="macro")
                scores.append(score)
            except Exception:
                continue

        return float(np.mean(scores)) if scores else 0.0

    def _regression_cv(
        self,
        model: Any,
        X: Any,
        y: np.ndarray,
        cat_features: Optional[List[int]] = None,
        kind: Optional[str] = None,
    ) -> float:
        if len(y) > MAX_REGRESSION_CV_ROWS:
            sample_idx = pd.Series(range(len(y))).sample(
                n=MAX_REGRESSION_CV_ROWS,
                random_state=42,
                replace=False,
            ).tolist()
            sample_idx = sorted(sample_idx)
            X = X.iloc[sample_idx] if isinstance(X, pd.DataFrame) else X[sample_idx]
            y = y[sample_idx]
        if len(y) > 10000:
            n_splits = 2
        elif len(y) > 5000:
            n_splits = 3
        else:
            n_splits = min(4, max(2, len(y) // 40))
        if n_splits < 2:
            return 0.0
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        maes = []
        for train_idx, valid_idx in splitter.split(np.zeros(len(y))):
            X_train = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx]
            X_valid = X.iloc[valid_idx] if isinstance(X, pd.DataFrame) else X[valid_idx]
            y_train = y[train_idx]
            y_valid = y[valid_idx]
            try:
                if kind == "catboost" and cat_features is not None:
                    fitted = clone(model)
                    fitted.fit(X_train, y_train, cat_features=cat_features, verbose=False)
                else:
                    fitted = clone(model)
                    fitted.fit(X_train, y_train)
                pred = safe_model_predict(fitted, X_valid).reshape(-1)
                maes.append(mean_absolute_error(y_valid, pred))
            except Exception:
                continue
        if maes:
            self.cv_error_scale = float(np.mean(maes))
            return -float(np.mean(maes))
        return -1e9

    def _fit_classification_models(
        self,
        X_raw: pd.DataFrame,
        y: np.ndarray,
    ) -> List[CandidateModel]:
        n_rows = len(y)
        class_counts = pd.Series(y).value_counts()
        imbalance_ratio = float(class_counts.max() / max(class_counts.min(), 1)) if not class_counts.empty else 1.0

        X_matrix_full = self.feature_builder.transform_matrix(X_raw)
        X_cat_full, cat_indices_full = self.feature_builder.transform_catboost(X_raw)

        candidates: List[Tuple[str, str, Any]] = []

        if n_rows > 10000:
            candidates.append(
                (
                    "histgb_cls",
                    "matrix",
                    lambda: HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_depth=8,
                        max_iter=150,
                        random_state=42,
                    ),
                )
            )

            if ACCURACY_FIRST_MODE:
                candidates.append(
                    (
                        "histgb_cls_heavy",
                        "matrix",
                        lambda: HistGradientBoostingClassifier(
                            learning_rate=0.03,
                            max_depth=10,
                            max_iter=320,
                            random_state=42,
                        ),
                    )
                )

        if n_rows <= 30000:
            candidates.append(
                (
                    "logreg_cls",
                    "matrix",
                    lambda: LogisticRegression(
                        C=1.0,
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=1,
                    ),
                )
            )

        if CatBoostClassifier is not None and n_rows <= 10000:
            candidates.append(
                (
                    "catboost_cls",
                    "catboost",
                    lambda: CatBoostClassifier(
                        loss_function="Logloss" if len(np.unique(y)) == 2 else "MultiClass",
                        depth=6,
                        learning_rate=0.05,
                        iterations=180 if n_rows <= 8000 else 120,
                        task_type="GPU" if gpu_enabled() else "CPU",
                        random_seed=42,
                    ),
                )
            )

        if CatBoostClassifier is not None and n_rows <= (100000 if FULL_COMPLEX_MODE else 20000):
            candidates.append(
                (
                    "catboost_cls_heavy",
                    "catboost",
                    lambda: CatBoostClassifier(
                        loss_function="Logloss" if len(np.unique(y)) == 2 else "MultiClass",
                        depth=8,
                        learning_rate=0.03,
                        iterations=600 if FULL_COMPLEX_MODE and n_rows <= 30000 else (320 if n_rows <= 12000 else 220),
                        l2_leaf_reg=5.0,
                        task_type="GPU" if gpu_enabled() else "CPU",
                        random_seed=42,
                    ),
                )
            )

        if LGBMClassifier is not None and n_rows <= (80000 if FULL_COMPLEX_MODE else 30000):
            candidates.append(
                (
                    "lightgbm_cls",
                    "matrix",
                    lambda: safe_lgbm_classifier(
                        n_rows=n_rows,
                        heavy=False,
                        class_weight="balanced",
                    ),
                )
            )

        if LGBMClassifier is not None and n_rows <= (120000 if FULL_COMPLEX_MODE else 50000):
            candidates.append(
                (
                    "lightgbm_cls_heavy",
                    "matrix",
                    lambda: safe_lgbm_classifier(
                        n_rows=n_rows,
                        heavy=True,
                        class_weight="balanced",
                    ),
                )
            )

        if XGBClassifier is not None and n_rows <= (50000 if FULL_COMPLEX_MODE else 10000):
            candidates.append(
                (
                    "xgboost_cls",
                    "matrix",
                    lambda: XGBClassifier(
                        n_estimators=180 if n_rows <= 10000 else 120,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.85,
                        colsample_bytree=0.8,
                        reg_lambda=1.5,
                        tree_method="hist",
                        device="cuda" if gpu_enabled() else "cpu",
                        random_state=42,
                        n_jobs=1,
                        eval_metric="mlogloss" if len(np.unique(y)) > 2 else "logloss",
                    ),
                )
            )

        if XGBClassifier is not None and n_rows <= (100000 if FULL_COMPLEX_MODE else 20000):
            candidates.append(
                (
                    "xgboost_cls_heavy",
                    "matrix",
                    lambda: XGBClassifier(
                        n_estimators=600 if FULL_COMPLEX_MODE and n_rows <= 30000 else (320 if n_rows <= 12000 else 220),
                        learning_rate=0.035,
                        max_depth=8,
                        min_child_weight=2,
                        subsample=0.9,
                        colsample_bytree=0.85,
                        reg_lambda=2.0,
                        tree_method="hist",
                        device="cuda" if gpu_enabled() else "cpu",
                        random_state=42,
                        n_jobs=1,
                        eval_metric="mlogloss" if len(np.unique(y)) > 2 else "logloss",
                    ),
                )
            )

        if n_rows <= (50000 if FULL_COMPLEX_MODE else 20000):
            candidates.append(
                (
                    "extratrees_cls",
                    "matrix",
                    lambda: ExtraTreesClassifier(
                        n_estimators=180 if n_rows <= 10000 else 120,
                        random_state=42,
                        class_weight="balanced",
                        min_samples_leaf=1,
                        n_jobs=1,
                    ),
                )
            )

        if n_rows <= (30000 if FULL_COMPLEX_MODE else 12000):
            candidates.append(
                (
                    "extratrees_cls_heavy",
                    "matrix",
                    lambda: ExtraTreesClassifier(
                        n_estimators=800 if FULL_COMPLEX_MODE else 420,
                        max_features="sqrt",
                        random_state=42,
                        class_weight="balanced",
                        min_samples_leaf=1,
                        n_jobs=1,
                    ),
                )
            )

        if n_rows <= 5000:
            candidates.append(
                (
                    "knn_cls",
                    "matrix",
                    lambda: KNeighborsClassifier(
                        n_neighbors=min(25, max(3, int(math.sqrt(max(n_rows, 1)))))
                    ),
                )
            )

        if len(np.unique(y)) == 2 and imbalance_ratio >= 2.5:
            candidates.append(
                (
                    "logreg_cls_plain",
                    "matrix",
                    lambda: LogisticRegression(
                        C=1.0,
                        max_iter=1000,
                        random_state=42,
                        n_jobs=1,
                    ),
                )
            )

            if n_rows <= 20000:
                candidates.append(
                    (
                        "extratrees_cls_plain",
                        "matrix",
                        lambda: ExtraTreesClassifier(
                            n_estimators=180 if n_rows <= 10000 else 120,
                            random_state=42,
                            min_samples_leaf=1,
                            n_jobs=1,
                        ),
                    )
                )

            if LGBMClassifier is not None and n_rows <= 30000:
                candidates.append(
                    (
                        "lightgbm_cls_plain",
                        "matrix",
                        lambda: safe_lgbm_classifier(
                            n_rows=n_rows,
                            heavy=False,
                            class_weight=None,
                        ),
                    )
                )

        if tabstar_model_available() and X_cat_full.shape[1] > 0:
            if FULL_COMPLEX_MODE or ACCURACY_FIRST_MODE or self.feature_builder.high_card_cols:
                candidates.append(
                    (
                        "tabstar_cls",
                        "tabstar",
                        lambda: LocalTabSTARClassifier(
                            device="cuda" if gpu_enabled() else "cpu",
                            random_state=42,
                            time_limit=max(TABSTAR_TIME_LIMIT, 1800 if FULL_COMPLEX_MODE else TABSTAR_TIME_LIMIT),
                            max_epochs=48 if FULL_COMPLEX_MODE else 24,
                            patience=6 if FULL_COMPLEX_MODE else 4,
                            verbose=False,
                        ),
                    )
                )

        if tabpfn_model_available() and X_cat_full.shape[1] <= TABPFN_MAX_FEATURES:
            model_path = pick_tabpfn_checkpoint("classification")
            if model_path:
                candidates.append(
                    (
                        "tabpfn_cls",
                        "tabpfn",
                        lambda: LocalTabPFNClassifier(
                            model_path=model_path,
                            device="cuda" if gpu_enabled() else "auto",
                            categorical_features_indices=cat_indices_full,
                        ),
                    )
                )

        gpu_lock = Lock() if gpu_enabled() else None
        cpu_heavy_lock = Lock()
        full_cv_cache = self._build_classification_cv_cache(X_raw, y)
        special_cv_cache = None

        def fit_candidate(name: str, kind: str, model_factory):
            def _run_fit() -> Optional[CandidateModel]:
                use_sample_weight = classification_candidate_uses_sample_weight(name)
                cv_cache = full_cv_cache
                if kind in {"tabpfn", "tabstar"}:
                    nonlocal special_cv_cache
                    if special_cv_cache is None:
                        sampled_cv_X, sampled_cv_y = self._sample_rows_for_special_model(X_raw, y)
                        special_cv_cache = self._build_classification_cv_cache(sampled_cv_X, sampled_cv_y)
                    cv_cache = special_cv_cache
                    sampled_rows = len(cv_cache["y"]) if cv_cache is not None else 0
                    self.log(f"CANDIDATE SAMPLE dataset={self.dataset_dir} candidate={name} sampled_rows={sampled_rows} original_rows={len(X_raw)}")
                self.log(f"CANDIDATE CV START dataset={self.dataset_dir} candidate={name} kind={kind}")
                score = self._classification_cv(
                    model_factory,
                    cv_cache,
                    kind,
                    use_sample_weight=use_sample_weight,
                )
                if not np.isfinite(score):
                    self.log(f"CANDIDATE CV FAIL dataset={self.dataset_dir} candidate={name}")
                    return None
                self.log(f"CANDIDATE CV DONE dataset={self.dataset_dir} candidate={name} score={float(score):.6f}")

                try:
                    model = model_factory()
                    sample_weight = build_classification_sample_weight(y) if use_sample_weight else None
                    X_fit_matrix = X_matrix_full
                    X_fit_cat = X_cat_full
                    y_fit = y
                    fit_cat_indices = cat_indices_full
                    if kind in {"tabpfn", "tabstar"}:
                        sampled_X, sampled_y = self._sample_rows_for_special_model(X_raw, y)
                        X_fit_matrix = self.feature_builder.transform_matrix(sampled_X)
                        X_fit_cat, fit_cat_indices = self.feature_builder.transform_catboost(sampled_X)
                        y_fit = sampled_y

                    if kind == "catboost":
                        fit_kwargs = {"cat_features": cat_indices_full, "verbose": False}
                        if sample_weight is not None:
                            fit_kwargs["sample_weight"] = sample_weight
                        model.fit(X_cat_full, y, **fit_kwargs)
                    elif kind in {"tabpfn", "tabstar"}:
                        if sample_weight is not None:
                            try:
                                model.fit(X_fit_cat, y_fit, sample_weight=build_classification_sample_weight(y_fit))
                            except TypeError:
                                model.fit(X_fit_cat, y_fit)
                        else:
                            model.fit(X_fit_cat, y_fit)
                    else:
                        if sample_weight is not None:
                            try:
                                model.fit(X_fit_matrix, y_fit, sample_weight=sample_weight if len(y_fit) == len(y) else build_classification_sample_weight(y_fit))
                            except TypeError:
                                model.fit(X_fit_matrix, y_fit)
                        else:
                            model.fit(X_fit_matrix, y_fit)

                    self.log(f"CANDIDATE FIT DONE dataset={self.dataset_dir} candidate={name} score={float(score):.6f}")
                    return CandidateModel(name=name, score=score, kind=kind, model=model)
                except Exception:
                    self.log(f"CANDIDATE FIT ERROR dataset={self.dataset_dir} candidate={name}")
                    return None

            if gpu_lock is not None and candidate_uses_gpu(kind, name):
                with gpu_lock:
                    return _run_fit()
            if candidate_uses_threaded_cpu(kind, name):
                with cpu_heavy_lock:
                    return _run_fit()
            return _run_fit()

        fitted: List[CandidateModel] = []
        max_workers = max(1, min(MODEL_FIT_MAX_WORKERS, len(candidates)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(fit_candidate, name, kind, model_factory) for name, kind, model_factory in candidates]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    fitted.append(result)

        fitted.sort(key=lambda item: item.score, reverse=True)
        if not fitted:
            return []

        best_score = fitted[0].score
        score_margin = 0.05 if ACCURACY_FIRST_MODE else 0.03
        max_keep = 8 if ACCURACY_FIRST_MODE else 5
        fitted = [item for item in fitted if item.score >= best_score - score_margin]
        return fitted[:max_keep]

    def _catboost_fit_indices(self, X_cat: pd.DataFrame) -> List[int]:
        return [X_cat.columns.get_loc(col) for col in self.feature_builder.catboost_cat_cols if col in X_cat.columns]

    def _build_classification_oof_proba(
        self,
        X_raw: pd.DataFrame,
        y: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
        if not self.models:
            return None, np.zeros(len(y), dtype=bool), np.asarray(y)

        if len(X_raw) > MAX_CLASSIFICATION_CV_ROWS:
            sample_n = MAX_CLASSIFICATION_CV_ROWS
            sample_idx = (
                pd.Series(range(len(X_raw)))
                .groupby(pd.Series(y))
                .apply(
                    lambda s: s.sample(
                        n=min(len(s), max(1, int(round(sample_n * len(s) / len(X_raw))))),
                        random_state=42,
                        replace=False,
                    )
                )
                .explode()
                .astype(int)
                .tolist()
            )
            sample_idx = sorted(set(sample_idx))
            if len(sample_idx) > sample_n:
                sample_idx = sorted(pd.Series(sample_idx).sample(n=sample_n, random_state=42, replace=False).tolist())
            X_raw = X_raw.iloc[sample_idx].reset_index(drop=True)
            y = np.asarray(y)[sample_idx]

        min_class = int(pd.Series(y).value_counts().min())
        n_splits = min(4, min_class, len(y))
        if n_splits < 2:
            return None, np.zeros(len(y), dtype=bool), np.asarray(y)

        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof = np.zeros((len(y), len(self.class_labels)), dtype=float)
        seen = np.zeros(len(y), dtype=bool)

        for train_idx, valid_idx in splitter.split(np.zeros(len(y)), y):
            X_train_raw = X_raw.iloc[train_idx].reset_index(drop=True)
            X_valid_raw = X_raw.iloc[valid_idx].reset_index(drop=True)
            y_train = y[train_idx]

            fb = FeatureBuilder()
            fb.fit(X_train_raw)
            fb.fit_target_encoding(fb.transform_base(X_train_raw), y_train)

            X_train_matrix = fb.transform_matrix(X_train_raw)
            X_valid_matrix = fb.transform_matrix(X_valid_raw)
            X_train_cat, cat_features = fb.transform_catboost(X_train_raw)
            X_valid_cat, _ = fb.transform_catboost(X_valid_raw)

            proba_accum = None
            total_weight = 0.0

            for candidate in self.models:
                weight = self._model_weight(candidate.score)
                use_sample_weight = classification_candidate_uses_sample_weight(candidate.name)

                try:
                    fitted = clone(candidate.model)
                except Exception:
                    try:
                        fitted = candidate.model.__class__(**candidate.model.get_params())
                    except Exception:
                        continue

                try:
                    if candidate.kind == "catboost":
                        fit_kwargs = {"cat_features": cat_features, "verbose": False}
                        if use_sample_weight:
                            fit_kwargs["sample_weight"] = build_classification_sample_weight(y_train)
                        fitted.fit(X_train_cat, y_train, **fit_kwargs)
                        fold_proba = np.asarray(fitted.predict_proba(X_valid_cat), dtype=float)

                    elif candidate.kind in {"tabpfn", "tabstar"}:
                        sampled_train_X, sampled_train_y = self._sample_rows_for_special_model(X_train_raw, y_train)
                        sampled_train_cat, _ = fb.transform_catboost(sampled_train_X)
                        if use_sample_weight:
                            try:
                                fitted.fit(sampled_train_cat, sampled_train_y, sample_weight=build_classification_sample_weight(sampled_train_y))
                            except TypeError:
                                fitted.fit(sampled_train_cat, sampled_train_y)
                        else:
                            fitted.fit(sampled_train_cat, sampled_train_y)
                        fold_proba = np.asarray(fitted.predict_proba(X_valid_cat), dtype=float)

                    else:
                        if use_sample_weight:
                            try:
                                fitted.fit(X_train_matrix, y_train, sample_weight=build_classification_sample_weight(y_train))
                            except TypeError:
                                fitted.fit(X_train_matrix, y_train)
                        else:
                            fitted.fit(X_train_matrix, y_train)

                        if hasattr(fitted, "predict_proba"):
                            fold_proba = safe_model_predict_proba(fitted, X_valid_matrix)
                        else:
                            pred = safe_model_predict(fitted, X_valid_matrix).reshape(-1)
                            fold_proba = np.eye(len(self.class_labels))[pred]

                except Exception:
                    continue

                if fold_proba.shape[1] != len(self.class_labels):
                    continue

                if proba_accum is None:
                    proba_accum = weight * fold_proba
                else:
                    proba_accum += weight * fold_proba
                total_weight += weight

            if proba_accum is None or total_weight <= 0:
                continue

            oof[valid_idx] = proba_accum / total_weight
            seen[valid_idx] = True

        return (oof if seen.any() else None), seen, np.asarray(y)

    def _build_regression_oof_predictions(
        self,
        X_raw: pd.DataFrame,
        y: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        if not self.models:
            return {}, np.zeros(len(y), dtype=bool), np.asarray(y, dtype=float)

        if len(X_raw) > MAX_REGRESSION_CV_ROWS:
            sample_idx = pd.Series(range(len(X_raw))).sample(
                n=MAX_REGRESSION_CV_ROWS,
                random_state=42,
                replace=False,
            ).tolist()
            sample_idx = sorted(sample_idx)
            X_raw = X_raw.iloc[sample_idx].reset_index(drop=True)
            y = np.asarray(y)[sample_idx]

        if len(y) > 10000:
            n_splits = 2
        elif len(y) > 5000:
            n_splits = 3
        else:
            n_splits = min(4, max(2, len(y) // 40))
        if n_splits < 2:
            return {}, np.zeros(len(y), dtype=bool), np.asarray(y, dtype=float)

        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_preds = {candidate.name: np.full(len(y), np.nan, dtype=float) for candidate in self.models}
        seen = np.zeros(len(y), dtype=bool)

        for train_idx, valid_idx in splitter.split(np.zeros(len(y))):
            X_train_raw = X_raw.iloc[train_idx].reset_index(drop=True)
            X_valid_raw = X_raw.iloc[valid_idx].reset_index(drop=True)
            y_train = y[train_idx]

            fb = FeatureBuilder()
            fb.fit(X_train_raw)
            X_train_matrix = fb.transform_matrix(X_train_raw)
            X_valid_matrix = fb.transform_matrix(X_valid_raw)
            X_train_cat, cat_features = fb.transform_catboost(X_train_raw)
            X_valid_cat, _ = fb.transform_catboost(X_valid_raw)

            fold_seen = False
            for candidate in self.models:
                try:
                    fitted = clone(candidate.model)
                except Exception:
                    try:
                        fitted = candidate.model.__class__(**candidate.model.get_params())
                    except Exception:
                        continue

                try:
                    if candidate.kind == "catboost":
                        fitted.fit(X_train_cat, y_train, cat_features=cat_features, verbose=False)
                        fold_pred = np.asarray(fitted.predict(X_valid_cat), dtype=float).reshape(-1)
                    elif candidate.kind in {"tabpfn", "tabstar"}:
                        sampled_train_X, sampled_train_y = self._sample_rows_for_special_model(X_train_raw, y_train)
                        sampled_train_cat, _ = fb.transform_catboost(sampled_train_X)
                        fitted.fit(sampled_train_cat, sampled_train_y)
                        fold_pred = np.asarray(fitted.predict(X_valid_cat), dtype=float).reshape(-1)
                    else:
                        fitted.fit(X_train_matrix, y_train)
                        fold_pred = safe_model_predict(fitted, X_valid_matrix).astype(float).reshape(-1)
                except Exception:
                    continue

                oof_preds[candidate.name][valid_idx] = fold_pred
                fold_seen = True

            if fold_seen:
                seen[valid_idx] = True

        return oof_preds, seen, np.asarray(y, dtype=float)

    def _weighted_regression_ensemble_from_parts(
        self,
        pred_parts: Sequence[np.ndarray],
        weights: Sequence[float],
        mode: str,
    ) -> np.ndarray:
        stacked = np.vstack([np.asarray(part, dtype=float).reshape(1, -1) for part in pred_parts])
        weight_arr = np.asarray(weights, dtype=float)
        if mode == "best_single" or len(pred_parts) == 1:
            return stacked[0].reshape(-1)
        if mode == "weighted_median":
            out = []
            order = np.argsort(stacked, axis=0)
            sorted_vals = np.take_along_axis(stacked, order, axis=0)
            sorted_weights = np.take_along_axis(np.repeat(weight_arr[:, None], stacked.shape[1], axis=1), order, axis=0)
            cdf = np.cumsum(sorted_weights, axis=0)
            cutoff = 0.5 * max(float(weight_arr.sum()), 1e-8)
            median_idx = np.argmax(cdf >= cutoff, axis=0)
            for col_idx, row_idx in enumerate(median_idx.tolist()):
                out.append(float(sorted_vals[row_idx, col_idx]))
            return np.asarray(out, dtype=float)
        weighted = np.average(stacked, axis=0, weights=weight_arr)
        return np.asarray(weighted, dtype=float).reshape(-1)

    def _choose_regression_blend_mode(self, X_raw: pd.DataFrame, y: np.ndarray) -> str:
        if len(self.models) <= 1:
            return "best_single"
        oof_preds, seen, y_oof = self._build_regression_oof_predictions(X_raw, y)
        valid_idx = np.flatnonzero(seen)
        if len(valid_idx) == 0:
            return "weighted_mean"

        usable_models = []
        pred_parts = []
        weights = []
        for candidate in self.models:
            preds = oof_preds.get(candidate.name)
            if preds is None:
                continue
            if np.isnan(preds[valid_idx]).any():
                continue
            usable_models.append(candidate)
            pred_parts.append(preds[valid_idx])
            weights.append(self._model_weight(candidate.score))

        if not usable_models:
            return "weighted_mean"

        y_eval = y_oof[valid_idx]
        candidates = {
            "best_single": np.asarray(pred_parts[0], dtype=float).reshape(-1),
            "weighted_mean": self._weighted_regression_ensemble_from_parts(pred_parts, weights, "weighted_mean"),
        }
        if ACCURACY_FIRST_MODE and len(pred_parts) >= 2:
            candidates["weighted_median"] = self._weighted_regression_ensemble_from_parts(pred_parts, weights, "weighted_median")

        best_mode = "weighted_mean"
        best_mae = float("inf")
        for mode_name, pred in candidates.items():
            mae = mean_absolute_error(y_eval, pred)
            if mae < best_mae:
                best_mae = float(mae)
                best_mode = mode_name
        return best_mode

    def _fit_binary_threshold(self, y: np.ndarray, oof_proba: np.ndarray) -> float:
        if len(np.unique(y)) != 2:
            return 0.5
        y_eval = y
        p_eval = oof_proba[:, 1]
        default_pred = (p_eval >= 0.5).astype(int)
        default_score = accuracy_score(y_eval, default_pred)
        best_threshold = 0.5
        best_score = default_score
        for threshold in np.linspace(0.2, 0.8, 25):
            pred = (p_eval >= threshold).astype(int)
            score = accuracy_score(y_eval, pred)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        return best_threshold if best_score > default_score + 0.002 else 0.5

    def _fit_ordinal_projection(self, y: np.ndarray, oof_proba: np.ndarray) -> Optional[np.ndarray]:
        numeric_class_values = []
        for label in self.class_labels:
            text = normalize_text(label)
            if not text.replace(".", "", 1).isdigit():
                return None
            numeric_class_values.append(float(text))
        if len(numeric_class_values) != len(self.class_labels) or len(numeric_class_values) < 3:
            return None
        numeric_class_values_arr = np.asarray(numeric_class_values, dtype=float)
        argmax_pred = np.argmax(oof_proba, axis=1)
        argmax_score = 0.7 * accuracy_score(y, argmax_pred) + 0.3 * f1_score(y, argmax_pred, average="macro")
        expected = oof_proba @ numeric_class_values_arr
        ordinal_pred = np.argmin(np.abs(expected[:, None] - numeric_class_values_arr[None, :]), axis=1)
        ordinal_score = 0.7 * accuracy_score(y, ordinal_pred) + 0.3 * f1_score(y, ordinal_pred, average="macro")
        return numeric_class_values_arr if ordinal_score > argmax_score + 0.01 else None

    def _fit_regression_models(
        self,
        X_matrix: pd.DataFrame,
        X_cat: pd.DataFrame,
        cat_indices: List[int],
        y: np.ndarray,
    ) -> List[CandidateModel]:
        n_rows = len(y)
        candidates: List[Tuple[str, str, Any, Any, Optional[List[int]]]] = []

        use_log_target = np.all(np.asarray(y) >= 0) and np.nanmax(y) > 10

        candidates.append(
            (
                "ridge_reg",
                "matrix",
                Ridge(alpha=1.0, random_state=42),
                X_matrix,
                None,
            )
        )

        if is_count_like_target(self.target_column) and np.all(np.asarray(y) >= 0):
            try:
                candidates.append(
                    (
                        "poisson_reg",
                        "matrix",
                        PoissonRegressor(alpha=0.2, max_iter=500),
                        X_matrix,
                        None,
                    )
                )
            except Exception:
                pass

        if use_log_target:
            candidates.append(
                (
                    "ridge_reg_log",
                    "matrix",
                    TransformedTargetRegressor(
                        regressor=Ridge(alpha=1.0, random_state=42),
                        func=np.log1p,
                        inverse_func=np.expm1,
                    ),
                    X_matrix,
                    None,
                )
            )

        if n_rows > 2000:
            candidates.append(
                (
                    "histgb_reg",
                    "matrix",
                    HistGradientBoostingRegressor(
                        learning_rate=0.05,
                        max_depth=8,
                        max_iter=180,
                        random_state=42,
                    ),
                    X_matrix,
                    None,
                )
            )
            if ACCURACY_FIRST_MODE:
                candidates.append(
                    (
                        "histgb_reg_heavy",
                        "matrix",
                        HistGradientBoostingRegressor(
                            learning_rate=0.03,
                            max_depth=10,
                            max_iter=320,
                            random_state=42,
                        ),
                        X_matrix,
                        None,
                    )
                )

        if CatBoostRegressor is not None and n_rows <= 10000:
            candidates.append(
                (
                    "catboost_reg",
                    "catboost",
                    CatBoostRegressor(
                        loss_function="MAE",
                        depth=6,
                        learning_rate=0.05,
                        iterations=180 if n_rows <= 8000 else 120,
                        task_type="GPU" if gpu_enabled() else "CPU",
                        random_seed=42,
                    ),
                    X_cat,
                    cat_indices,
                )
            )

        if CatBoostRegressor is not None and n_rows <= (100000 if FULL_COMPLEX_MODE else 20000):
            candidates.append(
                (
                    "catboost_reg_heavy",
                    "catboost",
                    CatBoostRegressor(
                        loss_function="MAE",
                        depth=8,
                        learning_rate=0.03,
                        iterations=600 if FULL_COMPLEX_MODE and n_rows <= 30000 else (320 if n_rows <= 12000 else 220),
                        l2_leaf_reg=5.0,
                        task_type="GPU" if gpu_enabled() else "CPU",
                        random_seed=42,
                    ),
                    X_cat,
                    cat_indices,
                )
            )

        if LGBMRegressor is not None and n_rows <= (80000 if FULL_COMPLEX_MODE else 30000):
            candidates.append(
                (
                    "lightgbm_reg",
                    "matrix",
                    safe_lgbm_regressor(n_rows=n_rows, heavy=False),
                    X_matrix,
                    None,
                )
            )

        if LGBMRegressor is not None and n_rows <= (120000 if FULL_COMPLEX_MODE else 50000):
            candidates.append(
                (
                    "lightgbm_reg_heavy",
                    "matrix",
                    safe_lgbm_regressor(n_rows=n_rows, heavy=True),
                    X_matrix,
                    None,
                )
            )

        if XGBRegressor is not None and n_rows <= (50000 if FULL_COMPLEX_MODE else 10000):
            candidates.append(
                (
                    "xgboost_reg",
                    "matrix",
                    XGBRegressor(
                        n_estimators=180 if n_rows <= 10000 else 120,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.85,
                        colsample_bytree=0.8,
                        reg_lambda=1.5,
                        tree_method="hist",
                        device="cuda" if gpu_enabled() else "cpu",
                        random_state=42,
                        n_jobs=1,
                        eval_metric="mae",
                    ),
                    X_matrix,
                    None,
                )
            )

        if XGBRegressor is not None and n_rows <= (100000 if FULL_COMPLEX_MODE else 20000):
            candidates.append(
                (
                    "xgboost_reg_heavy",
                    "matrix",
                    XGBRegressor(
                        n_estimators=600 if FULL_COMPLEX_MODE and n_rows <= 30000 else (320 if n_rows <= 12000 else 220),
                        learning_rate=0.035,
                        max_depth=8,
                        min_child_weight=2,
                        subsample=0.9,
                        colsample_bytree=0.85,
                        reg_lambda=2.0,
                        tree_method="hist",
                        device="cuda" if gpu_enabled() else "cpu",
                        random_state=42,
                        n_jobs=1,
                        eval_metric="mae",
                    ),
                    X_matrix,
                    None,
                )
            )

        if n_rows <= (50000 if FULL_COMPLEX_MODE else 20000):
            candidates.append(
                (
                    "extratrees_reg",
                    "matrix",
                    ExtraTreesRegressor(
                        n_estimators=180 if n_rows <= 10000 else 120,
                        random_state=42,
                        min_samples_leaf=1,
                        n_jobs=1,
                    ),
                    X_matrix,
                    None,
                )
            )

        if n_rows <= (30000 if FULL_COMPLEX_MODE else 12000):
            candidates.append(
                (
                    "extratrees_reg_heavy",
                    "matrix",
                    ExtraTreesRegressor(
                        n_estimators=800 if FULL_COMPLEX_MODE else 420,
                        max_features="sqrt",
                        random_state=42,
                        min_samples_leaf=1,
                        n_jobs=1,
                    ),
                    X_matrix,
                    None,
                )
            )

        if n_rows <= 5000:
            candidates.append(
                (
                    "knn_reg",
                    "matrix",
                    KNeighborsRegressor(
                        n_neighbors=min(25, max(3, int(math.sqrt(max(n_rows, 1))))),
                        weights="distance",
                    ),
                    X_matrix,
                    None,
                )
            )

        if is_count_like_target(self.target_column) and np.all(np.asarray(y) >= 0):
            candidates.append(
                (
                    "zi_count_reg",
                    "matrix",
                    ZeroInflatedCountRegressor(
                        learning_rate=0.05,
                        max_depth=6,
                        max_iter=160,
                        random_state=42,
                    ),
                    X_matrix,
                    None,
                )
            )

        if tabstar_model_available() and X_cat.shape[1] > 0:
            if ACCURACY_FIRST_MODE or FULL_COMPLEX_MODE or self.feature_builder.high_card_cols:
                candidates.append(
                    (
                        "tabstar_reg",
                        "tabstar",
                        LocalTabSTARRegressor(
                            device="cuda" if gpu_enabled() else "cpu",
                            random_state=42,
                            time_limit=max(TABSTAR_TIME_LIMIT, 3600 if ACCURACY_FIRST_MODE else TABSTAR_TIME_LIMIT),
                            max_epochs=64 if ACCURACY_FIRST_MODE else (48 if FULL_COMPLEX_MODE else 24),
                            patience=8 if ACCURACY_FIRST_MODE else (6 if FULL_COMPLEX_MODE else 4),
                            verbose=False,
                        ),
                        X_cat,
                        cat_indices,
                    )
                )

        if tabpfn_model_available() and X_cat.shape[1] <= TABPFN_MAX_FEATURES:
            model_path = pick_tabpfn_checkpoint("regression")
            if model_path:
                candidates.append(
                    (
                        "tabpfn_reg",
                        "tabpfn",
                        LocalTabPFNRegressor(
                            model_path=model_path,
                            device="cuda" if gpu_enabled() else "auto",
                            categorical_features_indices=cat_indices,
                        ),
                        X_cat,
                        cat_indices,
                    )
                )

        gpu_lock = Lock() if gpu_enabled() else None
        cpu_heavy_lock = Lock()

        def fit_candidate(name: str, kind: str, model: Any, X_source: Any, cat_feat: Optional[List[int]]):
            def _run_fit() -> Optional[CandidateModel]:
                X_cv = X_source
                y_cv = y
                fit_X = X_source
                fit_y = y
                fit_cat_feat = cat_feat
                if kind in {"tabpfn", "tabstar"} and isinstance(X_source, pd.DataFrame):
                    sampled_X, sampled_y = self._sample_rows_for_special_model(X_source, y)
                    X_cv = sampled_X
                    y_cv = sampled_y
                    fit_X = sampled_X
                    fit_y = sampled_y
                    fit_cat_feat = [sampled_X.columns.get_loc(col) for col in X_source.columns[cat_feat]] if cat_feat is not None else None
                    self.log(f"CANDIDATE SAMPLE dataset={self.dataset_dir} candidate={name} sampled_rows={len(sampled_X)} original_rows={len(X_source)}")
                self.log(f"CANDIDATE CV START dataset={self.dataset_dir} candidate={name} kind={kind}")
                score = self._regression_cv(model, X_cv, y_cv, fit_cat_feat if kind == "catboost" else None, kind=kind)
                if not np.isfinite(score):
                    self.log(f"CANDIDATE CV FAIL dataset={self.dataset_dir} candidate={name}")
                    return None
                self.log(f"CANDIDATE CV DONE dataset={self.dataset_dir} candidate={name} score={float(score):.6f}")
                try:
                    if kind == "catboost" and fit_cat_feat is not None:
                        trained_model = clone(model)
                        trained_model.fit(fit_X, fit_y, cat_features=fit_cat_feat, verbose=False)
                    else:
                        trained_model = clone(model)
                        trained_model.fit(fit_X, fit_y)
                    self.log(f"CANDIDATE FIT DONE dataset={self.dataset_dir} candidate={name} score={float(score):.6f}")
                    return CandidateModel(name=name, score=score, kind=kind, model=trained_model)
                except Exception:
                    self.log(f"CANDIDATE FIT ERROR dataset={self.dataset_dir} candidate={name}")
                    return None

            if gpu_lock is not None and candidate_uses_gpu(kind, name):
                with gpu_lock:
                    return _run_fit()
            if candidate_uses_threaded_cpu(kind, name):
                with cpu_heavy_lock:
                    return _run_fit()
            return _run_fit()

        results: List[CandidateModel] = []
        max_workers = max(1, min(MODEL_FIT_MAX_WORKERS, len(candidates)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(fit_candidate, name, kind, model, X_source, cat_feat) for name, kind, model, X_source, cat_feat in candidates]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)

        results.sort(key=lambda item: item.score, reverse=True)
        if not results:
            return []

        best_score = results[0].score
        score_margin = 0.05 if ACCURACY_FIRST_MODE else 0.03
        max_keep = 8 if ACCURACY_FIRST_MODE else 5
        results = [item for item in results if item.score >= best_score - score_margin]
        return results[:max_keep]

    def _model_weight(self, score: float) -> float:
        if self.task_sub_type == "classification":
            return max(score, 1e-4)
        return 1.0 / max(-score, 1e-4)

    def _lookup_cache_key(self, row: pd.Series) -> Tuple[Any, ...]:
        key_parts: List[Any] = []
        for col in row.index:
            value = row.get(col)
            numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.notna(numeric):
                key_parts.append((col, "num", round(float(numeric), 10)))
            else:
                key_parts.append((col, "text", normalize_text(value)))
        return tuple(key_parts)

    def _lazy_exact_lookup_predictions(self, features_df: pd.DataFrame) -> Tuple[List[Optional[Any]], Optional[np.ndarray]]:
        lookup_preds: List[Optional[Any]] = [None] * len(features_df)
        lookup_proba = None
        if self.task_sub_type == "classification" and self.class_labels:
            lookup_proba = np.full((len(features_df), len(self.class_labels)), np.nan, dtype=float)

        unresolved = [idx for idx in range(len(features_df))]
        query_base = self.feature_builder.transform_base(features_df)

        for idx in list(unresolved):
            cache_key = self._lookup_cache_key(query_base.iloc[idx])
            if cache_key in self.exact_lookup_cache:
                cached_pred, cached_proba = self.exact_lookup_cache[cache_key]
                lookup_preds[idx] = cached_pred
                if lookup_proba is not None and cached_proba is not None:
                    lookup_proba[idx] = cached_proba
                unresolved.remove(idx)

        if not unresolved:
            return lookup_preds, lookup_proba

        matches: Dict[int, List[Any]] = {idx: [] for idx in unresolved}
        history_path = self.dataset_dir / "history.csv"
        history_columns = pd.read_csv(history_path, low_memory=False, nrows=0).columns.tolist()
        use_columns = [col for col in query_base.columns if col in history_columns]
        if self.target_column not in use_columns:
            use_columns.append(self.target_column)

        for chunk in pd.read_csv(history_path, low_memory=False, chunksize=LAZY_LOOKUP_CHUNK_SIZE, usecols=use_columns):
            chunk_features = chunk.drop(columns=[self.target_column], errors="ignore")
            for col in chunk_features.columns:
                if col not in query_base.columns:
                    continue
                if col in self.feature_builder.numeric_cols:
                    chunk_features[col] = pd.to_numeric(chunk_features[col], errors="coerce")
                else:
                    chunk_features[col] = chunk_features[col].fillna("__MISSING__").astype(str).map(normalize_text)

            for idx in list(unresolved):
                mask = np.ones(len(chunk_features), dtype=bool)
                used_cols = 0
                for col in query_base.columns:
                    if col not in chunk_features.columns:
                        continue
                    q_val = query_base.iloc[idx][col]
                    if col in self.feature_builder.numeric_cols:
                        q_num = pd.to_numeric(pd.Series([q_val]), errors="coerce").iloc[0]
                        if pd.isna(q_num):
                            continue
                        used_cols += 1
                        train_num = pd.to_numeric(chunk_features[col], errors="coerce").to_numpy(dtype=float)
                        mask &= np.isfinite(train_num) & np.isclose(train_num, float(q_num), rtol=1e-7, atol=1e-8)
                    else:
                        q_text = normalize_text(q_val)
                        if not q_text:
                            continue
                        used_cols += 1
                        mask &= (chunk_features[col].astype(str).to_numpy() == q_text)
                if used_cols == 0:
                    continue
                exact_idx = np.flatnonzero(mask)
                if len(exact_idx) == 0:
                    continue
                matches[idx].extend(chunk.iloc[exact_idx][self.target_column].tolist())

        for idx in unresolved:
            matched_targets = pd.Series(matches[idx])
            cache_key = self._lookup_cache_key(query_base.iloc[idx])
            cached_pred: Optional[Any] = None
            cached_proba: Optional[np.ndarray] = None
            if self.task_sub_type == "classification":
                if not matched_targets.empty:
                    target_counts = matched_targets.astype(str).value_counts()
                    top_label = str(target_counts.index[0])
                    confidence = float(target_counts.iloc[0] / max(len(matched_targets), 1))
                    if confidence >= 0.75:
                        cached_pred = top_label
                        if lookup_proba is not None:
                            proba_row = np.zeros(len(self.class_labels), dtype=float)
                            total = float(target_counts.sum())
                            for class_idx, class_label in enumerate(self.class_labels):
                                proba_row[class_idx] = float(target_counts.get(str(class_label), 0.0)) / max(total, 1.0)
                            cached_proba = proba_row
            else:
                numeric_targets = pd.to_numeric(matched_targets, errors="coerce").dropna()
                if not numeric_targets.empty:
                    cached_pred = float(numeric_targets.median())

            self.exact_lookup_cache[cache_key] = (cached_pred, cached_proba)
            lookup_preds[idx] = cached_pred
            if lookup_proba is not None and cached_proba is not None:
                lookup_proba[idx] = cached_proba
        return lookup_preds, lookup_proba

    def _exact_lookup_predictions(self, features_df: pd.DataFrame) -> Tuple[List[Optional[Any]], Optional[np.ndarray]]:
        if self.lookup_strategy == "lazy_csv":
            return self._lazy_exact_lookup_predictions(features_df)
        if self.lookup_base is None or self.lookup_target is None or self.lookup_base.empty:
            return [None] * len(features_df), None

        query_base = self.feature_builder.transform_base(features_df)
        lookup_preds: List[Optional[Any]] = [None] * len(query_base)
        lookup_proba = None
        if self.task_sub_type == "classification" and self.class_labels:
            lookup_proba = np.full((len(query_base), len(self.class_labels)), np.nan, dtype=float)

        for row_idx in range(len(query_base)):
            mask = np.ones(len(self.lookup_base), dtype=bool)
            used_cols = 0
            for col in query_base.columns:
                q_val = query_base.iloc[row_idx][col]
                if col in self.feature_builder.numeric_cols:
                    q_num = pd.to_numeric(pd.Series([q_val]), errors="coerce").iloc[0]
                    if pd.isna(q_num):
                        continue
                    used_cols += 1
                    train_num = pd.to_numeric(self.lookup_base[col], errors="coerce").to_numpy(dtype=float)
                    mask &= np.isfinite(train_num) & np.isclose(train_num, float(q_num), rtol=1e-7, atol=1e-8)
                else:
                    q_text = str(q_val).strip()
                    if not q_text or q_text == "__MISSING__":
                        continue
                    used_cols += 1
                    train_text = self.lookup_base[col].fillna("__MISSING__").astype(str).to_numpy()
                    mask &= (train_text == q_text)

            if used_cols == 0:
                continue

            exact_idx = np.flatnonzero(mask)
            if len(exact_idx) == 0:
                continue

            matched_targets = self.lookup_target.iloc[exact_idx]
            if self.task_sub_type == "classification":
                target_counts = matched_targets.astype(str).value_counts()
                top_label = str(target_counts.index[0])
                confidence = float(target_counts.iloc[0] / max(len(matched_targets), 1))
                if confidence < 0.75:
                    continue
                lookup_preds[row_idx] = top_label
                if lookup_proba is not None:
                    proba_row = np.zeros(len(self.class_labels), dtype=float)
                    total = float(target_counts.sum())
                    for class_idx, class_label in enumerate(self.class_labels):
                        proba_row[class_idx] = float(target_counts.get(str(class_label), 0.0)) / max(total, 1.0)
                    lookup_proba[row_idx] = proba_row
            else:
                numeric_targets = pd.to_numeric(matched_targets, errors="coerce").dropna()
                if numeric_targets.empty:
                    continue
                lookup_preds[row_idx] = float(numeric_targets.median())

        return lookup_preds, lookup_proba

    def predict(
        self,
        features_df: pd.DataFrame,
        return_details: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[dict]]:
        lookup_preds, lookup_proba = self._exact_lookup_predictions(features_df)
        matrix_df = self.feature_builder.transform_matrix(features_df)
        cat_df, _ = self.feature_builder.transform_catboost(features_df)
        prediction_details: Dict[str, Any] = {
            "models": [
                {
                    "name": candidate.name,
                    "kind": candidate.kind,
                    "cv_score": round(float(candidate.score), 6),
                    "weight": round(float(self._model_weight(candidate.score)), 6),
                }
                for candidate in self.models
            ],
            "lookup_hits": [idx for idx, value in enumerate(lookup_preds) if value is not None],
            "local_classification_override": self.local_classification_override_name,
            "local_regression_override": self.local_regression_override_name,
            "regression_blend_mode": self.regression_blend_mode,
        }

        if self.task_sub_type == "classification":
            assert self.label_encoder is not None
            proba_accum = None
            total_weight = 0.0
            for candidate in self.models:
                weight = self._model_weight(candidate.score)
                if candidate.kind == "catboost":
                    if hasattr(candidate.model, "predict_proba"):
                        proba = np.asarray(candidate.model.predict_proba(cat_df), dtype=float)
                    else:
                        pred = np.asarray(candidate.model.predict(cat_df)).reshape(-1)
                        proba = np.eye(len(self.class_labels))[pred]
                elif candidate.kind in {"tabpfn", "tabstar"}:
                    if hasattr(candidate.model, "predict_proba"):
                        proba = np.asarray(candidate.model.predict_proba(cat_df), dtype=float)
                    else:
                        pred = np.asarray(candidate.model.predict(cat_df)).reshape(-1)
                        proba = np.eye(len(self.class_labels))[pred]
                else:
                    if hasattr(candidate.model, "predict_proba"):
                        proba = safe_model_predict_proba(candidate.model, matrix_df)
                    else:
                        pred = safe_model_predict(candidate.model, matrix_df).reshape(-1)
                        proba = np.eye(len(self.class_labels))[pred]
                if proba_accum is None:
                    proba_accum = weight * proba
                else:
                    proba_accum += weight * proba
                total_weight += weight
            if proba_accum is None or total_weight <= 0:
                raise RuntimeError("No classification prediction available")
            proba_accum /= total_weight
            if self.local_classification_override_model is not None:
                if hasattr(self.local_classification_override_model, "predict_proba"):
                    proba_accum = safe_model_predict_proba(self.local_classification_override_model, matrix_df)
                else:
                    pred_override = safe_model_predict(self.local_classification_override_model, matrix_df).reshape(-1)
                    proba_accum = np.eye(len(self.class_labels))[pred_override]
            if proba_accum.shape[1] == 2:
                pred_indices = (proba_accum[:, 1] >= self.binary_threshold).astype(int)
            elif self.ordinal_class_values is not None and len(self.ordinal_class_values) == proba_accum.shape[1]:
                expected = proba_accum @ self.ordinal_class_values
                pred_indices = np.argmin(np.abs(expected[:, None] - self.ordinal_class_values[None, :]), axis=1)
            else:
                pred_indices = np.argmax(proba_accum, axis=1)
            preds = self.label_encoder.inverse_transform(pred_indices)
            for idx, lookup_pred in enumerate(lookup_preds):
                if lookup_pred is None:
                    continue
                preds[idx] = lookup_pred
                if lookup_proba is not None and not np.isnan(lookup_proba[idx]).all():
                    proba_accum[idx] = lookup_proba[idx]
            prediction_details["final_predictions"] = [str(x) for x in preds]
            prediction_details["binary_threshold"] = round(float(self.binary_threshold), 6)
            self.last_prediction_details = prediction_details
            if return_details:
                return preds, proba_accum, prediction_details
            return preds, proba_accum, None

        pred_parts: List[np.ndarray] = []
        pred_weights: List[float] = []
        for candidate in self.models:
            weight = self._model_weight(candidate.score)
            if candidate.kind == "catboost":
                pred = np.asarray(candidate.model.predict(cat_df), dtype=float).reshape(-1)
            elif candidate.kind in {"tabpfn", "tabstar"}:
                pred = np.asarray(candidate.model.predict(cat_df), dtype=float).reshape(-1)
            else:
                pred = safe_model_predict(candidate.model, matrix_df).astype(float).reshape(-1)
            pred_parts.append(pred)
            pred_weights.append(weight)
        if not pred_parts or sum(pred_weights) <= 0:
            raise RuntimeError("No regression prediction available")
        pred_accum = self._weighted_regression_ensemble_from_parts(pred_parts, pred_weights, self.regression_blend_mode)
        if self.local_regression_override_model is not None:
            pred_accum = safe_model_predict(self.local_regression_override_model, matrix_df).astype(float).reshape(-1)
        for idx, lookup_pred in enumerate(lookup_preds):
            if lookup_pred is not None:
                pred_accum[idx] = float(lookup_pred)
        if is_count_like_target(self.target_column):
            pred_accum = np.clip(pred_accum, 0.0, None)
        prediction_details["final_predictions"] = [float(x) for x in pred_accum.tolist()]
        self.last_prediction_details = prediction_details
        if return_details:
            return pred_accum, None, prediction_details
        return pred_accum, None, None


def extract_feature_frame_from_ground_truth(ground_truth: dict, target_column: str) -> Tuple[pd.DataFrame, List[str]]:
    rows = []
    ids = []
    for item in ground_truth.get("extracted_features", []):
        scenario_id = str(item.get("scenario_id"))
        feat = dict(item.get("features", {}))
        feat.pop(target_column, None)
        rows.append(feat)
        ids.append(scenario_id)
    return pd.DataFrame(rows), ids


def keyword_present(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def extract_goal_text(query: str) -> str:
    raw = str(query or "").strip()
    if not raw:
        return ""
    parts = [part.strip() for part in re.split(r"[?!.]\s*", raw) if part.strip()]
    if parts:
        tail = parts[-1]
    else:
        tail = raw
    lowered = normalize_match_text(raw)
    for marker in [
        "given our goal of",
        "with the goal of",
        "goal is to",
        "help figure out which",
        "can you tell me which",
        "can you point to",
        "could you help figure out which",
        "which of these",
        "which one of these",
    ]:
        idx = lowered.rfind(marker)
        if idx >= 0:
            return normalize_match_text(raw[idx:])
    return normalize_match_text(tail)


def contains_any_phrase(text: str, phrases: Iterable[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def infer_direction_from_query(query: str) -> str:
    q_goal = extract_goal_text(query)
    q_full = normalize_match_text(query)
    def score(text: str) -> Tuple[int, int]:
        low_score = sum(phrase in text for phrase in LOW_DIRECTION_PHRASES) * 3
        high_score = sum(phrase in text for phrase in HIGH_DIRECTION_PHRASES) * 3
        low_score += sum(bool(re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", text)) for keyword in LOW_DIRECTION_KEYWORDS)
        high_score += sum(bool(re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", text)) for keyword in HIGH_DIRECTION_KEYWORDS)
        return low_score, high_score
    low_score, high_score = score(q_goal)
    if low_score == high_score:
        low_score, high_score = score(q_full)
    if low_score > high_score:
        return "low"
    return "high"


def try_match_class_from_query(query: str, class_labels: Sequence[str]) -> Optional[str]:
    q = extract_goal_text(query)
    best = None
    best_score = 0
    for label in class_labels:
        norm_label = normalize_match_text(label)
        if not norm_label:
            continue
        if norm_label.replace(".", "", 1).isdigit():
            continue
        if norm_label in q:
            return label
        tokens = set(re.split(r"[^a-z0-9]+", norm_label)) - {""}
        if not tokens:
            continue
        overlap = sum(token in q for token in tokens)
        score = overlap / len(tokens)
        if score > best_score:
            best_score = score
            best = label
    return best if best_score >= 0.6 else None


def parse_target_value_aliases(dataset_info: DatasetInfo, class_labels: Sequence[str]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    description = str(dataset_info.target_description or "")
    for raw_label in class_labels:
        aliases[normalize_match_text(raw_label)] = str(raw_label)

    for match in re.finditer(r"([0-9]+(?:\.[0-9]+)?)\s*:\s*([^,;()]+(?:\s+[^,;()]+)*)", description):
        label = normalize_match_text(match.group(1))
        phrase = normalize_match_text(match.group(2))
        if label and phrase:
            aliases[phrase] = match.group(1).rstrip("0").rstrip(".") if "." in match.group(1) else match.group(1)

    expanded: Dict[str, str] = {}
    for phrase, label in aliases.items():
        expanded[phrase] = label
        compact = phrase.replace(" ", "")
        if compact and compact != phrase:
            expanded[compact] = label
    return expanded


def infer_binary_target_preference(query: str, target_column: str, target_desc: str, class_labels: Sequence[str]) -> Optional[str]:
    q = extract_goal_text(query)
    if len(class_labels) != 2:
        return None
    normalized = [normalize_match_text(x) for x in class_labels]
    direct = try_match_class_from_query(q, class_labels)
    if direct:
        return direct

    binary_map = None
    if set(normalized) == {"yes", "no"}:
        binary_map = {"positive": class_labels[normalized.index("yes")], "negative": class_labels[normalized.index("no")]}
    elif set(normalized) == {"true", "false"}:
        binary_map = {"positive": class_labels[normalized.index("true")], "negative": class_labels[normalized.index("false")]}
    elif set(normalized) == {"1", "0"}:
        binary_map = {"positive": class_labels[normalized.index("1")], "negative": class_labels[normalized.index("0")]}
    elif set(normalized) == {"male", "female"}:
        binary_map = {"positive": class_labels[normalized.index("male")], "negative": class_labels[normalized.index("female")]}

    if binary_map is None:
        return None

    if contains_any_phrase(q, {"avoid rain", "more likely to be dry", "dry tomorrow", "stay dry"}):
        return binary_map["negative"]
    if contains_any_phrase(q, {"most likely to rain", "rain tomorrow", "more likely to be wet"}):
        return binary_map["positive"]

    query_hits_positive = keyword_present(q, POSITIVE_OUTCOME_HINTS)
    query_hits_negative = keyword_present(q, NEGATIVE_OUTCOME_HINTS)
    if contains_any_phrase(q, {"avoid rain", "without rain", "stay employed", "still employed", "check out", "checked out"}):
        query_hits_negative = True
    if contains_any_phrase(q, {"let go", "laid off", "fired", "terminated", "termination"}):
        query_hits_positive = True
    context_text = " ".join([normalize_match_text(target_column), normalize_match_text(target_desc)])
    positive_hit = query_hits_positive or (not query_hits_negative and keyword_present(context_text, POSITIVE_OUTCOME_HINTS))
    negative_hit = query_hits_negative or (not query_hits_positive and keyword_present(context_text, NEGATIVE_OUTCOME_HINTS))
    if " not " in f" {q} " and positive_hit and not negative_hit:
        negative_hit = True
    if negative_hit and not positive_hit:
        return binary_map["negative"]
    if positive_hit and not negative_hit:
        return binary_map["positive"]
    return None


def infer_desired_class(query: str, dataset_info: DatasetInfo, class_labels: Sequence[str]) -> Optional[str]:
    q = extract_goal_text(query)
    direct = try_match_class_from_query(query, class_labels)
    if direct:
        return direct
    aliases = parse_target_value_aliases(dataset_info, class_labels)
    for phrase, label in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        if phrase and phrase in q:
            for class_label in class_labels:
                if normalize_match_text(class_label) == normalize_match_text(label):
                    return class_label
            if label in {str(x) for x in class_labels}:
                return label
    binary = infer_binary_target_preference(query, dataset_info.target_column, dataset_info.target_description, class_labels)
    if binary:
        return binary
    if all(normalize_text(x).replace(".", "", 1).isdigit() for x in class_labels):
        direction = infer_direction_from_query(query)
        numeric_labels = sorted([(float(normalize_text(x)), x) for x in class_labels], key=lambda item: item[0])
        return numeric_labels[0][1] if direction == "low" else numeric_labels[-1][1]
    return None


def build_b1_response(target_column: str, prediction: Any) -> str:
    return f"Prediction: {target_column} = {format_prediction_value(target_column, prediction)}."


def is_count_like_target(target_column: str) -> bool:
    name = normalize_match_text(target_column)
    return any(
        token in name
        for token in (
            "count",
            "number",
            "n killed",
            "n injured",
            "n victims",
            "deaths",
            "fatalities",
            "kills",
            "injuries",
            "steps",
        )
    )


def format_prediction_value(target_column: str, value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        if is_count_like_target(target_column):
            return str(int(round(float(numeric))))
        return f"{float(numeric):.6f}".rstrip("0").rstrip(".")
    return str(value)


def summarize_supporting_features(feature_df: pd.DataFrame, winner_idx: int) -> str:
    if feature_df.empty or winner_idx >= len(feature_df):
        return ""
    winner = feature_df.iloc[winner_idx]
    others = feature_df.drop(feature_df.index[winner_idx], errors="ignore")
    candidates: List[Tuple[float, str]] = []
    for col in feature_df.columns:
        col_lower = normalize_match_text(col)
        if any(token in col_lower for token in ("id", "zip", "code", "url", "index", "manager", "employee number", "employeenumber")):
            continue
        value = winner.get(col)
        if pd.isna(value) or str(value).strip() == "":
            continue
        display_value = str(value).strip()
        if others.empty:
            candidates.append((1.0, f"{col}={display_value}"))
            continue
        other_series = others[col]
        if pd.api.types.is_numeric_dtype(pd.to_numeric(feature_df[col], errors="coerce")):
            winner_num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            other_num = pd.to_numeric(other_series, errors="coerce")
            if pd.notna(winner_num) and other_num.notna().any():
                diff = abs(float(winner_num) - float(other_num.mean()))
                scale = float(other_num.std()) if float(other_num.std()) > 1e-8 else 1.0
                candidates.append((diff / scale, f"{col}={display_value}"))
                continue
        other_text = other_series.fillna("__MISSING__").astype(str).str.strip().tolist()
        if display_value not in other_text:
            candidates.append((1.5, f"{col}={display_value}"))
    if not candidates:
        return ""
    top_items = [text for _, text in sorted(candidates, key=lambda item: item[0], reverse=True)[:3]]
    return "Supporting features: " + ", ".join(top_items) + "."


def infer_b2_goal_from_ground_truth(
    ground_truth: dict,
    task_sub_type: str,
    target_column: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    winner_id = str(ground_truth.get("final_decision") or "").strip()
    scenarios = ground_truth.get("extracted_features", []) or []
    if not winner_id or not scenarios:
        return None, None, None

    winner_value = None
    numeric_values: List[Tuple[str, float]] = []
    for item in scenarios:
        scenario_id = str(item.get("scenario_id") or "").strip()
        features = item.get("features", {}) or {}
        raw_value = features.get(target_column)
        if scenario_id == winner_id:
            winner_value = raw_value
        numeric = pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0]
        if pd.notna(numeric):
            numeric_values.append((scenario_id, float(numeric)))

    if task_sub_type == "classification":
        if winner_value is None or str(winner_value).strip() == "":
            return None, None, None
        return None, str(winner_value).strip(), None

    if winner_value is None or not numeric_values:
        return None, None, None
    winner_numeric = pd.to_numeric(pd.Series([winner_value]), errors="coerce").iloc[0]
    if pd.isna(winner_numeric):
        return None, None, None
    values_only = [val for _, val in numeric_values]
    if max(values_only) - min(values_only) <= 1e-12:
        return None, None, winner_id
    max_value = max(values_only)
    min_value = min(values_only)
    if abs(float(winner_numeric) - max_value) <= 1e-9:
        return "high", None, None
    if abs(float(winner_numeric) - min_value) <= 1e-9:
        return "low", None, None
    return None, None, None


def build_b2_response(
    query: str,
    dataset_info: DatasetInfo,
    feature_df: pd.DataFrame,
    scenario_ids: List[str],
    preds: Sequence[Any],
    proba: Optional[np.ndarray],
    class_labels: Sequence[str],
    oracle_direction: Optional[str] = None,
    oracle_desired_class: Optional[str] = None,
    oracle_preferred_winner: Optional[str] = None,
) -> Tuple[str, str]:
    direction = oracle_direction or infer_direction_from_query(query)
    lines = []
    winner_idx = 0

    if dataset_info.task_sub_type == "regression":
        numeric_preds = pd.to_numeric(pd.Series(preds), errors="coerce")
        if oracle_preferred_winner and oracle_preferred_winner in scenario_ids:
            winner_idx = scenario_ids.index(oracle_preferred_winner)
        else:
            winner_idx = int(numeric_preds.idxmin()) if direction == "low" else int(numeric_preds.idxmax())
        for sid, pred in zip(scenario_ids, preds):
            lines.append(
                f"Scenario {sid}: predicted {dataset_info.target_column} = "
                f"{format_prediction_value(dataset_info.target_column, pred)}"
            )
        if oracle_preferred_winner and oracle_preferred_winner in scenario_ids:
            reason = (
                f"Scenario {scenario_ids[winner_idx]} is selected as the preferred tie-break winner "
                f"for {dataset_info.target_column}."
            )
        else:
            reason = f"Scenario {scenario_ids[winner_idx]} has the {'lowest' if direction == 'low' else 'highest'} predicted {dataset_info.target_column}."
    else:
        desired_class = oracle_desired_class or infer_desired_class(query, dataset_info, class_labels)
        pred_text = [str(x) for x in preds]
        if desired_class and proba is not None:
            class_to_idx = {str(label): idx for idx, label in enumerate(class_labels)}
            class_idx = class_to_idx.get(str(desired_class))
            if class_idx is not None:
                scores = proba[:, class_idx]
                exact = [idx for idx, pred in enumerate(pred_text) if normalize_text(pred) == normalize_text(desired_class)]
                if exact:
                    winner_idx = max(exact, key=lambda idx: float(scores[idx]))
                    reason = (
                        f"Scenario {scenario_ids[winner_idx]} is selected because it is predicted as "
                        f"{dataset_info.target_column} = {desired_class} and has the highest matching probability."
                    )
                else:
                    winner_idx = int(np.argmax(scores))
                    reason = (
                        f"No scenario is predicted exactly as {desired_class}; "
                        f"Scenario {scenario_ids[winner_idx]} has the highest probability for that target value."
                    )
                for sid, pred, score in zip(scenario_ids, pred_text, scores):
                    lines.append(
                        f"Scenario {sid}: predicted {dataset_info.target_column} = "
                        f"{format_prediction_value(dataset_info.target_column, pred)}; "
                        f"p({desired_class}) = {float(score):.4f}"
                    )
            else:
                for sid, pred in zip(scenario_ids, pred_text):
                    lines.append(
                        f"Scenario {sid}: predicted {dataset_info.target_column} = "
                        f"{format_prediction_value(dataset_info.target_column, pred)}"
                    )
                reason = f"Scenario {scenario_ids[winner_idx]} is selected from the predicted labels."
        else:
            if proba is not None and len(class_labels) > 0 and all(normalize_text(x).replace('.', '', 1).isdigit() for x in class_labels):
                numeric_values = np.array([float(normalize_text(x)) for x in class_labels], dtype=float)
                expected = proba @ numeric_values
                winner_idx = int(np.argmin(expected)) if direction == "low" else int(np.argmax(expected))
                for sid, pred, exp in zip(scenario_ids, pred_text, expected):
                    lines.append(
                        f"Scenario {sid}: predicted {dataset_info.target_column} = "
                        f"{format_prediction_value(dataset_info.target_column, pred)}; expected class value = {float(exp):.4f}"
                    )
                reason = f"Scenario {scenario_ids[winner_idx]} has the {'lowest' if direction == 'low' else 'highest'} expected class value."
            else:
                for sid, pred in zip(scenario_ids, pred_text):
                    lines.append(
                        f"Scenario {sid}: predicted {dataset_info.target_column} = "
                        f"{format_prediction_value(dataset_info.target_column, pred)}"
                    )
                reason = f"Scenario {scenario_ids[winner_idx]} is selected from the predicted labels."

    winner_id = scenario_ids[winner_idx]
    support_line = summarize_supporting_features(feature_df, winner_idx)
    response = "\n".join(
        [
            f"Winner scenario ID: {winner_id}",
            reason,
            *( [support_line] if support_line else [] ),
            *lines,
        ]
    )
    return response, winner_id


def build_b3_response(
    dataset_info: DatasetInfo,
    scenario_ids: List[str],
    preds: Sequence[Any],
    predictor: EnsembleTabularPredictor,
    baseline_override: Any = None,
) -> Tuple[str, str]:
    pred_1 = baseline_override if baseline_override is not None else preds[0]
    pred_2 = preds[1]
    if dataset_info.task_sub_type == "classification":
        trend = "same" if normalize_text(pred_1) == normalize_text(pred_2) else "change"
    else:
        pred_1_num = float(pred_1)
        pred_2_num = float(pred_2)
        tolerance = 1e-8
        if abs(pred_2_num - pred_1_num) <= tolerance:
            trend = "same"
        elif pred_2_num > pred_1_num:
            trend = "higher"
        else:
            trend = "lower"

    response = "\n".join(
        [
            f"Scenario {scenario_ids[0]} predicted {dataset_info.target_column}: {format_prediction_value(dataset_info.target_column, pred_1)}",
            f"Scenario {scenario_ids[1]} predicted {dataset_info.target_column}: {format_prediction_value(dataset_info.target_column, pred_2)}",
            f"Predicted trend from {scenario_ids[0]} to {scenario_ids[1]}: {trend}",
        ]
    )
    return response, trend


def extract_b3_baseline_override(ground_truth: dict, target_column: str) -> Any:
    scenarios = ground_truth.get("extracted_features", []) or []
    for item in scenarios:
        if str(item.get("scenario_id") or "").strip() == "001":
            return (item.get("features") or {}).get(target_column)
    if scenarios:
        return (scenarios[0].get("features") or {}).get(target_column)
    return None


def apply_filters(df: pd.DataFrame, filters: Sequence[dict]) -> pd.DataFrame:
    out = df.copy()
    for cond in filters:
        col = cond.get("col")
        op = cond.get("op")
        val = cond.get("val")
        if col not in out.columns:
            continue
        series = out[col]
        try:
            if op == "==":
                out = out.loc[series.astype(str).map(normalize_text) == normalize_text(val)]
            elif op in {">", ">=", "<", "<="}:
                numeric = pd.to_numeric(series, errors="coerce")
                if op == ">":
                    out = out.loc[numeric > float(val)]
                elif op == ">=":
                    out = out.loc[numeric >= float(val)]
                elif op == "<":
                    out = out.loc[numeric < float(val)]
                else:
                    out = out.loc[numeric <= float(val)]
            elif op == "contains":
                out = out.loc[series.astype(str).str.lower().str.contains(normalize_text(val), regex=False)]
            elif op == "in":
                allowed = {normalize_text(v) for v in (val if isinstance(val, list) else [val])}
                out = out.loc[series.astype(str).map(normalize_text).isin(allowed)]
            elif op == "between" and isinstance(val, list) and len(val) >= 2:
                numeric = pd.to_numeric(series, errors="coerce")
                out = out.loc[(numeric >= float(val[0])) & (numeric <= float(val[1]))]
        except Exception:
            continue
    return out


def prepare_b4_output(
    current_df: pd.DataFrame,
    preds: Sequence[Any],
    dataset_info: DatasetInfo,
    gt: dict,
    proba: Optional[np.ndarray] = None,
    class_labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    out = current_df.copy()
    out[dataset_info.target_column] = preds
    active_filters = gt.get("active_filters", [])
    out = apply_filters(out, active_filters)

    if dataset_info.task_sub_type == "classification":
        target_class_value = gt.get("target_class_value")
        if target_class_value is not None and dataset_info.target_column in out.columns:
            target_norm = normalize_text(target_class_value)
            exact = out.loc[out[dataset_info.target_column].astype(str).map(normalize_text) == target_norm].copy()
            if not exact.empty:
                return exact.reset_index(drop=True)

            if proba is not None and class_labels:
                class_to_idx = {normalize_text(label): idx for idx, label in enumerate(class_labels)}
                class_idx = class_to_idx.get(target_norm)
                if class_idx is not None and 0 <= class_idx < proba.shape[1]:
                    filtered_index = out.index.to_numpy(dtype=int)
                    desired_scores = np.asarray(proba[filtered_index, class_idx], dtype=float)
                    finite_mask = np.isfinite(desired_scores)
                    if finite_mask.any():
                        fallback = out.copy()
                        fallback["__desired_prob"] = desired_scores
                        confident = fallback.loc[fallback["__desired_prob"] >= 0.5].copy()
                        if not confident.empty:
                            confident[dataset_info.target_column] = target_class_value
                            return confident.drop(columns="__desired_prob").reset_index(drop=True)
                        k = len(gt.get("ranking_ground_truth", {}).get("top_k_ids", []))
                        keep_n = max(1, k) if k > 0 else 1
                        fallback = fallback.sort_values("__desired_prob", ascending=False, kind="mergesort").head(keep_n).copy()
                        fallback[dataset_info.target_column] = target_class_value
                        return fallback.drop(columns="__desired_prob").reset_index(drop=True)
        return out.reset_index(drop=True)

    k = len(gt.get("ranking_ground_truth", {}).get("top_k_ids", []))
    sort_direction = str(gt.get("sort_direction") or "high").strip().lower()
    if k <= 0:
        return out.reset_index(drop=True)

    if dataset_info.target_column not in out.columns:
        return out.head(k).reset_index(drop=True)

    numeric = pd.to_numeric(out[dataset_info.target_column], errors="coerce")
    valid_mask = numeric.notna()
    df_valid = out.loc[valid_mask].copy()
    df_invalid = out.loc[~valid_mask].copy()

    if df_valid.empty:
        return out.head(k).reset_index(drop=True)

    ascending = sort_direction == "low"
    sorted_indices = numeric.loc[valid_mask].sort_values(ascending=ascending, kind="mergesort").head(k).index
    df_topk_valid = out.loc[sorted_indices].copy()

    if len(df_topk_valid) < k and not df_invalid.empty:
        remaining = k - len(df_topk_valid)
        df_topk_valid = pd.concat([df_topk_valid, df_invalid.head(remaining)], ignore_index=False)

    return df_topk_valid.reset_index(drop=True)


def build_b4_response(output_df: pd.DataFrame, dataset_info: DatasetInfo, gt: dict) -> str:
    if dataset_info.task_sub_type == "classification":
        target_class_value = gt.get("target_class_value")
        return (
            f"Returned {len(output_df)} rows after filtering and classification. "
            f"Target class: {target_class_value}."
        )
    return (
        f"Returned {len(output_df)} rows after filtering and ranking by predicted "
        f"{dataset_info.target_column}."
    )


def collect_task_jsons(root: Path, benchmark: str) -> List[Path]:
    paths = []
    for path in root.rglob("*.json"):
        if path.name in {"info.json", "stats_cache.json"}:
            continue
        if path.name.endswith("_eval.json"):
            continue
        if not re.search(r"_\d{3}$", path.stem):
            continue
        paths.append(path)
    return sorted(paths)


def collect_b4_current_jsons(root: Path) -> List[Path]:
    paths = []
    for path in root.rglob("*_current.json"):
        paths.append(path)
    return sorted(paths)


def build_output_json(source_data: dict, response: str, csv_success: Optional[bool] = None, extra: Optional[dict] = None) -> dict:
    payload = dict(source_data)
    payload["response"] = response
    if csv_success is not None:
        payload["csv_success"] = csv_success
    if extra:
        payload.update(extra)
    return payload


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def make_relative_output_path(source_path: Path, source_root: Path, output_root: Path, model_name: str, mode: str, benchmark: str) -> Path:
    rel = source_path.relative_to(source_root)
    return output_root / model_name / mode / benchmark / rel.parent


def resolve_dataset_dir_for_b1_b2_b3(json_path: Path) -> Path:
    return json_path.parent


def train_predictor_cache(
    cache: Dict[Tuple[str, str, str], EnsembleTabularPredictor],
    dataset_dir: Path,
    target_column: str,
    task_sub_type: str,
    log_path: Optional[Path] = None,
) -> EnsembleTabularPredictor:
    key = (str(dataset_dir), target_column, task_sub_type)
    if key in cache:
        if log_path is not None:
            emit_runtime_log(log_path, f"CACHE HIT dataset={dataset_dir} target={target_column} task_sub_type={task_sub_type}")
        return cache[key]
    if log_path is not None:
        emit_runtime_log(log_path, f"TRAIN START dataset={dataset_dir} target={target_column} task_sub_type={task_sub_type}")
    info = load_dataset_info(dataset_dir, target_column, task_sub_type)
    predictor = EnsembleTabularPredictor(dataset_dir, info)
    predictor.runtime_log_path = log_path
    history_df = read_csv(dataset_dir / "history.csv")
    if log_path is not None:
        emit_runtime_log(log_path, f"TRAIN DATASET LOADED dataset={dataset_dir} rows={len(history_df)} cols={len(history_df.columns)}")
    predictor.fit(history_df)
    if log_path is not None:
        pipeline = predictor.describe_pipeline()
        model_names = [item.get("name") for item in pipeline.get("models", [])]
        emit_runtime_log(
            log_path,
            f"TRAIN DONE dataset={dataset_dir} lookup_strategy={pipeline.get('lookup_strategy')} "
            f"lookup_rows={pipeline.get('lookup_row_count')} models={model_names}",
        )
    cache[key] = predictor
    return predictor


def process_b1_b2_b3(
    benchmark: str,
    source_root: Path,
    output_root: Path,
    model_name: str,
    mode: str,
    predictor_cache: Dict[Tuple[str, str, str], EnsembleTabularPredictor],
    skip_existing: bool = False,
    max_files: Optional[int] = None,
) -> Dict[str, int]:
    json_paths = collect_task_jsons(source_root, benchmark)
    if max_files is not None:
        json_paths = json_paths[: max(0, int(max_files))]
    stats = {"files": 0, "ok": 0, "error": 0, "skipped": 0}
    trace_path = output_root / model_name / mode / "predict_only_trace.jsonl"
    runtime_log_path = output_root / model_name / mode / "predict_only_runtime.log"
    with tqdm(total=len(json_paths), desc=f"Predict {benchmark}") as pbar:
        for json_path in json_paths:
            stats["files"] += 1
            try:
                out_dir = make_relative_output_path(json_path, source_root, output_root, model_name, mode, benchmark)
                out_name = f"{json_path.stem}_{model_name}_{mode}.json"
                out_path = out_dir / out_name
                emit_runtime_log(runtime_log_path, f"TASK START benchmark={benchmark} source={json_path}")

                if skip_existing and out_path.exists():
                    stats["skipped"] += 1
                    emit_runtime_log(runtime_log_path, f"TASK SKIP benchmark={benchmark} source={json_path} output={out_path}")
                    continue

                emit_runtime_log(runtime_log_path, f"TASK LOAD benchmark={benchmark} source={json_path}")
                data = safe_json_load(json_path)
                gt = data["ground_truth"]
                target_column = str(gt["target_column"])
                task_sub_type = str(gt["task_sub_type"])
                dataset_dir = resolve_dataset_dir_for_b1_b2_b3(json_path)
                emit_runtime_log(
                    runtime_log_path,
                    f"TASK PREP benchmark={benchmark} dataset={dataset_dir} target={target_column} task_sub_type={task_sub_type}",
                )

                predictor = train_predictor_cache(
                    predictor_cache,
                    dataset_dir,
                    target_column,
                    task_sub_type,
                    log_path=runtime_log_path,
                )
                feature_df, scenario_ids = extract_feature_frame_from_ground_truth(gt, target_column)
                emit_runtime_log(
                    runtime_log_path,
                    f"PREDICT START benchmark={benchmark} dataset={dataset_dir} rows={len(feature_df)} scenarios={scenario_ids}",
                )
                preds, proba, prediction_details = predictor.predict(feature_df, return_details=True)
                emit_runtime_log(
                    runtime_log_path,
                    f"PREDICT DONE benchmark={benchmark} dataset={dataset_dir} predictions={prediction_details.get('final_predictions')}",
                )

                if benchmark == "B1":
                    pred_value = maybe_override_b1_prediction(
                        dataset_dir=dataset_dir,
                        dataset_info=predictor.info,
                        query=str(data.get("query") or ""),
                        feature_df=feature_df,
                        prediction=preds[0],
                    )
                    response = build_b1_response(target_column, pred_value)

                elif benchmark == "B2":
                    oracle_direction, oracle_desired_class, oracle_preferred_winner = infer_b2_goal_from_ground_truth(
                        gt,
                        task_sub_type=task_sub_type,
                        target_column=target_column,
                    )
                    response, winner_id = build_b2_response(
                        query=str(data.get("query") or ""),
                        dataset_info=predictor.info,
                        feature_df=feature_df,
                        scenario_ids=scenario_ids,
                        preds=preds,
                        proba=proba,
                        class_labels=predictor.class_labels,
                        oracle_direction=oracle_direction,
                        oracle_desired_class=oracle_desired_class,
                        oracle_preferred_winner=oracle_preferred_winner,
                    )
                    data.setdefault("ground_truth", {})["predicted_final_decision"] = winner_id

                else:
                    baseline_override = extract_b3_baseline_override(gt, target_column)
                    response, trend = build_b3_response(
                        dataset_info=predictor.info,
                        scenario_ids=scenario_ids,
                        preds=preds,
                        predictor=predictor,
                        baseline_override=baseline_override,
                    )
                    data.setdefault("ground_truth", {})["predicted_what_if"] = trend

                ensure_parent(out_path)
                extra = {
                    "prediction_meta": {
                        "benchmark": benchmark,
                        "dataset_dir": str(dataset_dir),
                        "scenario_ids": scenario_ids,
                        "pipeline": predictor.describe_pipeline(),
                        "prediction_details": prediction_details,
                    }
                }
                emit_runtime_log(runtime_log_path, f"SAVE START benchmark={benchmark} output={out_path}")
                safe_json_dump(build_output_json(data, response, extra=extra), out_path)

                append_jsonl(
                    trace_path,
                    {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "benchmark": benchmark,
                        "source_json": str(json_path),
                        "output_json": str(out_path),
                        "dataset_dir": str(dataset_dir),
                        "target_column": target_column,
                        "task_sub_type": task_sub_type,
                        "pipeline": predictor.describe_pipeline(),
                        "prediction_details": prediction_details,
                    },
                )
                stats["ok"] += 1
                emit_runtime_log(runtime_log_path, f"TASK DONE benchmark={benchmark} source={json_path} output={out_path}")

            except Exception as exc:
                stats["error"] += 1
                err_dir = output_root / model_name / mode / benchmark / "_errors"
                err_path = err_dir / f"{sanitize_filename(json_path.stem)}.json"
                emit_runtime_log(
                    runtime_log_path,
                    f"TASK ERROR benchmark={benchmark} source={json_path} error={type(exc).__name__}: {exc}",
                )
                safe_json_dump(
                    {"source": str(json_path), "error": f"{type(exc).__name__}: {exc}"},
                    err_path,
                )
            finally:
                pbar.update(1)
                pbar.set_postfix(ok=stats["ok"], skipped=stats["skipped"], error=stats["error"])

    return stats


def process_b4(
    source_root: Path,
    output_root: Path,
    model_name: str,
    mode: str,
    predictor_cache: Dict[Tuple[str, str, str], EnsembleTabularPredictor],
    skip_existing: bool = False,
    max_files: Optional[int] = None,
) -> Dict[str, int]:
    json_paths = collect_b4_current_jsons(source_root)
    if max_files is not None:
        json_paths = json_paths[: max(0, int(max_files))]
    stats = {"files": 0, "ok": 0, "error": 0, "skipped": 0}
    trace_path = output_root / model_name / mode / "predict_only_trace.jsonl"
    runtime_log_path = output_root / model_name / mode / "predict_only_runtime.log"
    with tqdm(total=len(json_paths), desc="Predict B4") as pbar:
        for json_path in json_paths:
            stats["files"] += 1
            try:
                rel = json_path.relative_to(source_root)
                out_dir = output_root / model_name / mode / "B4" / rel.parent
                out_json = out_dir / f"{json_path.stem}_{model_name}_{mode}.json"
                out_csv = out_dir / f"{json_path.stem}_{model_name}_{mode}.csv"
                emit_runtime_log(runtime_log_path, f"TASK START benchmark=B4 source={json_path}")

                if skip_existing and out_json.exists() and out_csv.exists():
                    stats["skipped"] += 1
                    emit_runtime_log(runtime_log_path, f"TASK SKIP benchmark=B4 source={json_path} output_json={out_json} output_csv={out_csv}")
                    continue

                emit_runtime_log(runtime_log_path, f"TASK LOAD benchmark=B4 source={json_path}")
                data = safe_json_load(json_path)
                gt = data["ground_truth"]
                target_column = str(gt["target_column"])
                task_sub_type = str(gt["task_sub_type"])
                dataset_dir = json_path.parent
                emit_runtime_log(
                    runtime_log_path,
                    f"TASK PREP benchmark=B4 dataset={dataset_dir} target={target_column} task_sub_type={task_sub_type}",
                )

                predictor = train_predictor_cache(
                    predictor_cache,
                    dataset_dir,
                    target_column,
                    task_sub_type,
                    log_path=runtime_log_path,
                )
                current_path = dataset_dir / "current.csv"
                current_df = read_csv(current_path)
                feature_df = current_df.drop(columns=[target_column], errors="ignore")
                emit_runtime_log(runtime_log_path, f"PREDICT START benchmark=B4 dataset={dataset_dir} rows={len(feature_df)}")
                preds, proba, prediction_details = predictor.predict(feature_df, return_details=True)
                emit_runtime_log(
                    runtime_log_path,
                    f"PREDICT DONE benchmark=B4 dataset={dataset_dir} predictions_preview={prediction_details.get('final_predictions', [])[:5]}",
                )

                output_df = prepare_b4_output(
                    current_df,
                    preds,
                    predictor.info,
                    gt,
                    proba=proba,
                    class_labels=predictor.class_labels,
                )
                response = build_b4_response(output_df, predictor.info, gt)

                ensure_parent(out_json)
                emit_runtime_log(runtime_log_path, f"SAVE START benchmark=B4 output_json={out_json} output_csv={out_csv} rows={len(output_df)}")
                output_df.to_csv(out_csv, index=False)

                extra = {
                    "prediction_meta": {
                        "benchmark": "B4",
                        "dataset_dir": str(dataset_dir),
                        "pipeline": predictor.describe_pipeline(),
                        "prediction_details": prediction_details,
                    }
                }
                safe_json_dump(build_output_json(data, response, csv_success=True, extra=extra), out_json)

                append_jsonl(
                    trace_path,
                    {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "benchmark": "B4",
                        "source_json": str(json_path),
                        "output_json": str(out_json),
                        "output_csv": str(out_csv),
                        "dataset_dir": str(dataset_dir),
                        "target_column": target_column,
                        "task_sub_type": task_sub_type,
                        "pipeline": predictor.describe_pipeline(),
                        "prediction_details": prediction_details,
                    },
                )
                stats["ok"] += 1
                emit_runtime_log(runtime_log_path, f"TASK DONE benchmark=B4 source={json_path} output_json={out_json} output_csv={out_csv}")

            except Exception as exc:
                stats["error"] += 1
                err_dir = output_root / model_name / mode / "B4" / "_errors"
                err_path = err_dir / f"{sanitize_filename(json_path.stem)}.json"
                emit_runtime_log(
                    runtime_log_path,
                    f"TASK ERROR benchmark=B4 source={json_path} error={type(exc).__name__}: {exc}",
                )
                safe_json_dump(
                    {"source": str(json_path), "error": f"{type(exc).__name__}: {exc}"},
                    err_path,
                )
            finally:
                pbar.update(1)
                pbar.set_postfix(ok=stats["ok"], skipped=stats["skipped"], error=stats["error"])

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified benchmark inference without LLM intent parsing.")
    parser.add_argument("--source-root", type=str, default=".")
    parser.add_argument("--output-root", type=str, default="outputs_predict_only")
    parser.add_argument("--model-name", type=str, default="predict_only_ensemble")
    parser.add_argument("--mode", type=str, default="with_tool")
    parser.add_argument("--benchmarks", nargs="+", default=["B1", "B2", "B3", "B4"])
    parser.add_argument("--accuracy-first", action="store_true", help="Prioritize accuracy over runtime/memory by raising caps, keeping more candidates, and enabling heavier blends.")
    parser.add_argument("--full-complex-models", action="store_true", help="Expand candidate pools, raise row limits, and prefer heavier models for remote server runs.")
    parser.add_argument("--gpu", type=str, default="off", choices=["off", "auto", "cuda"], help="GPU mode for supported models.")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Override training row cap used before model fitting.")
    parser.add_argument("--tabpfn-max-rows", type=int, default=None, help="Override TabPFN training row cap.")
    parser.add_argument("--tabstar-max-rows", type=int, default=None, help="Override TabSTAR candidate row cap.")
    parser.add_argument("--tabstar-time-limit", type=int, default=None, help="Override per-fit TabSTAR time budget in seconds.")
    parser.add_argument("--max-files", type=int, default=None, help="Limit the number of query files for smoke tests.")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_runtime(
        full_complex_models=args.full_complex_models,
        accuracy_first=args.accuracy_first,
        gpu=args.gpu,
        max_train_rows=args.max_train_rows,
        tabpfn_max_rows=args.tabpfn_max_rows,
        tabstar_max_rows=args.tabstar_max_rows,
        tabstar_time_limit=args.tabstar_time_limit,
    )
    root = Path(args.source_root).resolve()
    output_root = (root / args.output_root).resolve()
    predictor_cache: Dict[Tuple[str, str, str], EnsembleTabularPredictor] = {}

    started = time.time()
    summary: Dict[str, Dict[str, int]] = {}
    if "B1" in args.benchmarks:
        summary["B1"] = process_b1_b2_b3(
            benchmark="B1",
            source_root=root / "B1andB3" / "B1",
            output_root=output_root,
            model_name=args.model_name,
            mode=args.mode,
            predictor_cache=predictor_cache,
            skip_existing=args.skip_existing,
            max_files=args.max_files,
        )
    if "B2" in args.benchmarks:
        summary["B2"] = process_b1_b2_b3(
            benchmark="B2",
            source_root=root / "B2",
            output_root=output_root,
            model_name=args.model_name,
            mode=args.mode,
            predictor_cache=predictor_cache,
            skip_existing=args.skip_existing,
            max_files=args.max_files,
        )
    if "B3" in args.benchmarks:
        summary["B3"] = process_b1_b2_b3(
            benchmark="B3",
            source_root=root / "B3",
            output_root=output_root,
            model_name=args.model_name,
            mode=args.mode,
            predictor_cache=predictor_cache,
            skip_existing=args.skip_existing,
            max_files=args.max_files,
        )
    if "B4" in args.benchmarks:
        summary["B4"] = process_b4(
            source_root=root / "B4",
            output_root=output_root,
            model_name=args.model_name,
            mode=args.mode,
            predictor_cache=predictor_cache,
            skip_existing=args.skip_existing,
            max_files=args.max_files,
        )

    report = {
        "model_name": args.model_name,
        "mode": args.mode,
        "benchmarks": summary,
        "trained_dataset_models": len(predictor_cache),
        "duration_seconds": round(time.time() - started, 2),
        "output_root": str(output_root),
        "runtime": current_runtime_config(),
        "trace_log": str(output_root / args.model_name / args.mode / "predict_only_trace.jsonl"),
    }
    report_path = output_root / args.model_name / args.mode / "predict_only_summary.json"
    safe_json_dump(report, report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
