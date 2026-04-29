import os
import json
import pandas as pd
import numpy as np
import argparse
import logging
import traceback
import fnmatch
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
# [新增] 引入 scipy 用于计算排序相关性
from scipy.stats import kendalltau

# ================= 日志配置 =================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation_optimized.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
DISABLE_STATS_CACHE = False

# ================= 基础工具 =================
def safe_json_load(filepath: str) -> Optional[Dict]:
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"读取JSON失败 {filepath}: {e}")
        return None

def safe_json_dump(data: Any, filepath: str, **kwargs) -> bool:
    try:
        kwargs.setdefault('indent', 2)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, **kwargs)
        return True
    except Exception as e:
        logger.error(f"写入JSON失败 {filepath}: {e}")
        return False

# ================= 向量化工具类 (性能核心) =================
class VectorizedUtils:
    @staticmethod
    def clean_series_vectorized(series: pd.Series) -> pd.Series:
        str_cleaned = series.astype(str).str.strip().str.lower()
        nums = pd.to_numeric(series, errors='coerce')
        mask_num = nums.notna() 
        
        if not mask_num.any():
            return str_cleaned

        valid_nums = nums[mask_num].astype(float)
        rounded = np.round(valid_nums)
        is_int = np.abs(valid_nums - rounded) < 1e-6
        
        ints_formatted = rounded.astype(int).astype(str)
        floats_formatted = np.round(valid_nums, 4).astype(str)
        
        final_nums = np.where(is_int, ints_formatted, floats_formatted)
        
        result = str_cleaned.copy()
        result.loc[mask_num] = final_nums
        return result

    @staticmethod
    def normalize_dataframe(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        if df.empty: return df
        df_out = df.copy()
        df_out.columns = [str(c).strip() for c in df_out.columns]
        valid_cols = [c for c in cols if c in df_out.columns]
        for col in valid_cols:
            df_out[col] = VectorizedUtils.clean_series_vectorized(df_out[col])
        return df_out

# ================= 统计缓存管理器 =================
class StatsCache:
    @staticmethod
    def compute_and_cache_stats(base_dir: str) -> Dict:
        cache_path = os.path.join(base_dir, "stats_cache.json")
        cached = safe_json_load(cache_path)
        if cached:
            first_numeric = next((v for v in cached.values() if v.get("type") == "numeric"), None)
            if first_numeric and "q95" in first_numeric:
                return cached

        if DISABLE_STATS_CACHE:
            logger.warning(f"统计缓存缺失或不完整，且已禁止重算: {cache_path}")
            return {}

        
        dfs = []
        candidate_files = ["history.csv", "train.csv", "test.csv"]
        found_any = False
        
        for fname in candidate_files:
            fpath = os.path.join(base_dir, fname)
            if os.path.exists(fpath):
                try:
                    df_temp = pd.read_csv(fpath, on_bad_lines='skip', encoding_errors='ignore', low_memory=False)
                    dfs.append(df_temp)
                    found_any = True
                except Exception as e:
                    logger.warning(f"读取CSV失败 {fpath}: {e}")
        
        if not found_any or not dfs:
            return {}
        
        full_df = pd.concat(dfs, ignore_index=True)
        stats = {}
        
        for col in full_df.columns:
            series = full_df[col]
            clean_series = series.dropna()
            if clean_series.empty: continue
            
            col_stat = {}
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                numeric_series = clean_series.astype(float)
                desc = numeric_series.describe()
                col_stat["type"] = "numeric"
                col_stat["std"] = float(desc.get("std", 0.0))
                col_stat["mean"] = float(desc.get("mean", 0.0))
                col_stat["min"] = float(desc.get("min", 0.0))
                col_stat["max"] = float(desc.get("max", 0.0))
                try:
                    q_values = numeric_series.quantile([0.05, 0.95]).to_dict()
                    col_stat["q05"] = float(q_values.get(0.05, col_stat["min"]))
                    col_stat["q95"] = float(q_values.get(0.95, col_stat["max"]))
                except Exception as e:
                    logger.warning(f"列 {col} 分位数计算失败: {e}")
                    col_stat["q05"] = col_stat["min"]
                    col_stat["q95"] = col_stat["max"]
                
                unique_vals = numeric_series.unique()
                if len(unique_vals) < 50:
                    col_stat["discrete_values"] = sorted([float(x) for x in unique_vals])
            else:
                unique_vals = clean_series.unique()
                n_unique = len(unique_vals)
                n_total = len(clean_series)
                if pd.api.types.is_bool_dtype(series):
                    col_stat["type"] = "categorical"
                    col_stat["categories"] = ["false", "true"]
                    col_stat["cardinality"] = 2
                elif n_unique == n_total or (n_total > 0 and n_unique / n_total > 0.8):
                    col_stat["type"] = "string"
                    col_stat["cardinality"] = n_unique
                else:
                    col_stat["type"] = "categorical"
                    col_stat["categories"] = sorted([str(x) for x in unique_vals])
                    col_stat["cardinality"] = n_unique
            stats[col] = col_stat
        if not DISABLE_STATS_CACHE:
            safe_json_dump(stats, cache_path)
        return stats

# ================= 筛选引擎 =================
class FilterEngine:
    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: List[Dict]) -> Tuple[pd.DataFrame, float]:
        if not filters or df.empty: return df, 1.0
        temp_df = df.copy()
        for col in temp_df.columns:
            s_numeric = pd.to_numeric(temp_df[col], errors='coerce')
            mask_original_valid = temp_df[col].notna()
            mask_converted_nan = s_numeric.isna()
            if (mask_original_valid & mask_converted_nan).sum() == 0:
                temp_df[col] = s_numeric

        mask = pd.Series(True, index=temp_df.index)
        for f in filters:
            col = f.get('col')
            op = f.get('op')
            val = f.get('val')
            if col not in temp_df.columns: continue
            s = temp_df[col]
            is_num = pd.api.types.is_numeric_dtype(s)
            try:
                if op == '==':
                    if is_num: mask &= np.isclose(s, float(val), atol=1e-5)
                    else: mask &= (s.astype(str).str.lower() == str(val).lower())
                elif op in ['>', '>=', '<', '<=']:
                    s_num = pd.to_numeric(s, errors='coerce')
                    val_float = float(val)
                    if op == '>': mask &= (s_num > val_float)
                    elif op == '>=': mask &= (s_num >= val_float)
                    elif op == '<': mask &= (s_num < val_float)
                    elif op == '<=': mask &= (s_num <= val_float)
                elif op == 'contains':
                    mask &= s.astype(str).str.lower().str.contains(str(val).lower(), regex=False)
                elif op == 'in':
                    vals = val if isinstance(val, list) else [val]
                    if is_num:
                        f_vals = [float(v) for v in vals if str(v).replace('.','',1).isdigit()]
                        mask &= s.isin(f_vals)
                    else:
                        s_vals = [str(v).lower() for v in vals]
                        mask &= s.astype(str).str.lower().isin(s_vals)
                elif op == 'between':
                    if isinstance(val, list) and len(val) >= 2:
                        s_num = pd.to_numeric(s, errors='coerce')
                        mask &= (s_num >= float(val[0])) & (s_num <= float(val[1]))
            except Exception: pass
        filtered_df = df.loc[mask].copy()
        compliance = len(filtered_df) / len(df) if len(df) > 0 else 1.0
        return filtered_df, compliance

# ================= 指标计算器 =================
class MetricCalculator:
    @staticmethod
    def calc_set_metrics(df_pred_norm: pd.DataFrame, df_gt_norm: pd.DataFrame, match_cols: List[str]) -> Tuple[Dict, List[Tuple[int, int]]]:
        if df_gt_norm.empty: return {"recall": 1.0, "precision": 1.0, "f1": 1.0}, []
        if df_pred_norm.empty: return {"recall": 0.0, "precision": 0.0, "f1": 0.0}, []

        p_work = df_pred_norm.reset_index().rename(columns={'index': 'orig_idx'})
        g_work = df_gt_norm.reset_index().rename(columns={'index': 'orig_idx'})
        
        p_filled = p_work[match_cols].fillna("<<NAN>>")
        g_filled = g_work[match_cols].fillna("<<NAN>>")

        p_work['cc'] = p_filled.groupby(match_cols).cumcount()
        g_work['cc'] = g_filled.groupby(match_cols).cumcount()
        
        p_work_merge = pd.concat([p_work, p_filled.add_suffix('_key')], axis=1)
        g_work_merge = pd.concat([g_work, g_filled.add_suffix('_key')], axis=1)
        
        merge_on_cols = [f"{c}_key" for c in match_cols] + ['cc']
        merged = pd.merge(p_work_merge, g_work_merge, on=merge_on_cols, how='inner', suffixes=('_p', '_g'))
        
        match_pairs = list(zip(merged['orig_idx_p'], merged['orig_idx_g']))
        tp = len(match_pairs)
        fp = len(df_pred_norm) - tp
        fn = len(df_gt_norm) - tp
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        if len(df_pred_norm) == 0 and len(df_gt_norm) == 0: precision, recall = 1.0, 1.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"recall": recall, "precision": precision, "f1": f1}, match_pairs

    @staticmethod
    def calc_nmae(pred_df_raw: pd.DataFrame, gt_df_raw: pd.DataFrame, match_pairs: List[Tuple[int, int]], target_col: str, dataset_stats: Dict) -> float:
        """
        修改后的 NMAE：不再考虑未匹配到的 GT，直接对匹配到的样本取归一化误差的平均值。
        """
        if gt_df_raw.empty: return 0.0
        if not match_pairs: return 1.0 # 如果一个都没匹配到，认为误差最大

        t_stat = dataset_stats.get(target_col, {})
        # 使用 min-max 归一化，避免分位数截断掩盖真实误差
        d_min = t_stat.get("min", 0)
        d_max = t_stat.get("max", 0)
        denom = (d_max - d_min) if (d_max - d_min) > 1e-5 else abs(t_stat.get("mean", 1.0))
        if denom < 1e-5: denom = 1.0

        matched_error_sum = 0.0
        if match_pairs and target_col in pred_df_raw.columns and target_col in gt_df_raw.columns:
            p_idxs, g_idxs = zip(*match_pairs)
            p_vals = pd.to_numeric(pred_df_raw.loc[list(p_idxs), target_col], errors='coerce').fillna(0.0).values
            g_vals = pd.to_numeric(gt_df_raw.loc[list(g_idxs), target_col], errors='coerce').fillna(0.0).values
            abs_diff = np.abs(p_vals - g_vals)
            norm_diff = np.minimum(abs_diff / denom, 1.0)
            matched_error_sum = np.sum(norm_diff)
            # 修改点：直接计算匹配到的取平均即可
            return min(matched_error_sum / len(match_pairs), 1.0)
        
        # 兜底：如果 target_col 缺失但有匹配对，认为匹配到的这些行误差为 1
        return 1.0

    @staticmethod
    def calc_ndcg(pred_df_sorted: pd.DataFrame, gt_df_raw: pd.DataFrame, match_pairs: List[Tuple[int, int]], target_col: str, k: int, sort_direction: str = 'high') -> float:
        if gt_df_raw.empty: return 1.0
        if pred_df_sorted.empty: return 0.0
        gt_series = pd.to_numeric(gt_df_raw[target_col], errors='coerce').fillna(0.0)
        if sort_direction == 'low': gt_series = -gt_series
        all_gt_values = gt_series.values
        gt_vals_dict = gt_series.to_dict()
        if len(all_gt_values) == 0: return 0.0
        min_val = np.min(all_gt_values)
        offset = abs(min_val) if min_val < 0 else 0.0
        def get_relevance(val): return val + offset
        pred_to_gt = dict(match_pairs)
        rel_scores = []
        for i, p_idx in enumerate(pred_df_sorted.index):
            if i >= k: break
            g_idx = pred_to_gt.get(p_idx)
            rel = get_relevance(gt_vals_dict.get(g_idx, 0.0)) if g_idx is not None else 0.0
            rel_scores.append(rel)
        if len(rel_scores) < k: rel_scores.extend([0.0] * (k - len(rel_scores)))
        rel_scores = np.array(rel_scores)
        discounts = np.log2(np.arange(len(rel_scores)) + 2)
        dcg = np.sum(rel_scores / discounts)
        ideal_rels = np.sort(get_relevance(all_gt_values))[::-1][:k]
        idcg_discounts = np.log2(np.arange(len(ideal_rels)) + 2)
        idcg = np.sum(ideal_rels / idcg_discounts)
        return dcg / idcg if idcg > 0 else 0.0
    
    # [新增] 计算 Kendall's Tau 秩相关系数
    @staticmethod
    def calc_kendall_tau(pred_df_raw: pd.DataFrame, gt_df_raw: pd.DataFrame, match_pairs: List[Tuple[int, int]], target_col: str) -> float:
        """
        计算预测值与真实值之间的秩相关系数 (Kendall's Tau)。
        仅计算匹配成功的行，反映模型对相对顺序的学习能力。
        """
        if not match_pairs or target_col not in pred_df_raw.columns or target_col not in gt_df_raw.columns:
            return 0.0
        
        p_idxs, g_idxs = zip(*match_pairs)
        
        # 提取匹配对的数值
        p_vals = pd.to_numeric(pred_df_raw.loc[list(p_idxs), target_col], errors='coerce').fillna(0.0).values
        g_vals = pd.to_numeric(gt_df_raw.loc[list(g_idxs), target_col], errors='coerce').fillna(0.0).values
        
        if len(p_vals) < 2:
            return 0.0 # 样本太少无法计算相关性
            
        try:
            tau, _ = kendalltau(p_vals, g_vals)
            return tau if not np.isnan(tau) else 0.0
        except Exception:
            return 0.0

# ================= 单任务逻辑 =================
def evaluate_single_task(model_dir: str, dataset_dir: str, task_name: str, model_name: str, mode_name: str) -> Dict:
    # 1. 定位 Dataset
    gt_csv_path = os.path.join(dataset_dir, "test_current.csv")
    # if not os.path.exists(gt_csv_path): gt_csv_path = os.path.join(dataset_dir, "test_current.csv")
    if not os.path.exists(gt_csv_path): return {"error": f"GT_CSV_MISSING: {gt_csv_path}"}

    # 2. Stats
    try: dataset_stats = StatsCache.compute_and_cache_stats(dataset_dir)
    except Exception as e: 
        print(f"Error: {e}")
        dataset_stats = {}

    # 3. 定位 Model Output
    try:
        files = os.listdir(model_dir)
        json_candidates = [f for f in files if f.endswith(".json") and "current" in f and model_name in f]
        if not json_candidates: json_candidates = [f for f in files if f.endswith(".json") and "current" in f]
        json_filename = json_candidates[0] if json_candidates else None
        
        csv_candidates = [f for f in files if f.endswith(".csv") and "current" in f and model_name in f]
        if not csv_candidates: csv_candidates = [f for f in files if f.endswith(".csv") and "current" in f]
        csv_filename = csv_candidates[0] if csv_candidates else None
    except Exception as e: return {"error": f"FILE_SEARCH_ERROR: {e}"}

    if not json_filename: return {"error": "MODEL_JSON_MISSING"}
    if not csv_filename: return {"error": "MODEL_CSV_MISSING"}

    json_path = os.path.join(model_dir, json_filename)
    model_csv_path = os.path.join(model_dir, csv_filename)

    # 4. 读取 Metadata
    meta = safe_json_load(json_path)
    if not meta: return {"error": "META_JSON_INVALID"}
    
    gt_info = meta.get("ground_truth", {})
    task_type = gt_info.get("task_sub_type", "regression")
    target_col = gt_info.get("target_column")
    active_filters = gt_info.get("active_filters", [])
    target_class_val = gt_info.get("target_class_value") 
    sort_dir = gt_info.get("sort_direction", "high") 

    # 5. 数据加载与评估
    is_csv_parse_error = False
    try:
        df_gt_raw = pd.read_csv(gt_csv_path, on_bad_lines='skip', low_memory=False).reset_index(drop=True)
        df_gt_raw.columns = [str(c).strip() for c in df_gt_raw.columns]
        
        try:
            df_model_raw = pd.read_csv(model_csv_path, on_bad_lines='skip', low_memory=False).reset_index(drop=True)
            df_model_raw.columns = [str(c).strip() for c in df_model_raw.columns]
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            logger.warning(f"CSV Parse Error in {task_name}: {e}")
            df_model_raw = pd.DataFrame()
            is_csv_parse_error = True
        except Exception as e:
            if "No columns to parse" in str(e) or "EmptyDataError" in str(e):
                logger.warning(f"CSV Parse Error (Generic) in {task_name}: {e}")
                df_model_raw = pd.DataFrame()
                is_csv_parse_error = True
            else:
                raise e 

    except Exception as e:
        return {"error": f"CSV_READ_ERROR: {e}"}

    is_csv_empty = df_model_raw.empty
    
    # [新增统计] Schema 一致性检查
    cols_match_history = False
    has_extra_cols = False
    has_missing_cols = False
    
    # 检查历史列名匹配 (现有逻辑)
    history_csv_path = os.path.join(dataset_dir, "history.csv")
    if os.path.exists(history_csv_path):
        try:
            df_hist_header = pd.read_csv(history_csv_path, nrows=0)
            hist_cols = set([str(c).strip() for c in df_hist_header.columns])
            model_cols = set(df_model_raw.columns)
            cols_match_history = (hist_cols == model_cols)
        except Exception: cols_match_history = False
    
    # [新增统计] 检查与 GT 列名的差异 (幻觉列/缺失列)
    gt_cols_set = set(df_gt_raw.columns)
    model_cols_set = set(df_model_raw.columns)
    has_extra_cols = len(model_cols_set - gt_cols_set) > 0 # 模型输出了GT里没有的列
    has_missing_cols = len(gt_cols_set - model_cols_set) > 0 # 模型遗漏了GT里有的列

    metrics = {
        "task_type": task_type, 
        "filter_compliance": 1.0,
        "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "nmae": 0.0, "ndcg": 0.0, "final_score": 0.0,
        "kendall_tau": 0.0, # 新增
        "is_csv_empty": is_csv_empty,
        "is_csv_parse_error": is_csv_parse_error,
        "cols_match_history": cols_match_history,
        "has_extra_cols": has_extra_cols, # 新增
        "has_missing_cols": has_missing_cols, # 新增
        "is_result_empty_after_filter": False, # 新增
        "is_filter_compliant": False,
        "is_pure_target_class": None,
        "is_exact_topk_count": None
    }

    match_cols = [c for c in df_gt_raw.columns if str(c).lower() != str(target_col).lower()]
    valid_match_cols = [c for c in match_cols if c in df_model_raw.columns]
    
    if not valid_match_cols and not is_csv_parse_error and not df_model_raw.empty:
        return {"error": "NO_MATCHING_COLS", "available": list(df_model_raw.columns)}

    if active_filters:
        df_model_filtered, compliance = FilterEngine.apply_filters(df_model_raw, active_filters)
        metrics["filter_compliance"] = compliance
        metrics["is_filter_compliant"] = (compliance == 1.0)
    else:
        df_model_filtered = df_model_raw.copy()
        metrics["is_filter_compliant"] = True 

    # [新增统计] 检查过滤后结果是否为空 (但CSV本身不为空)
    if df_model_filtered.empty and not is_csv_empty:
        metrics["is_result_empty_after_filter"] = True

    if task_type == "classification":
        is_pure = False
        if target_class_val is not None and target_col in df_model_filtered.columns:
            t_val = str(target_class_val).lower().strip()
            actual_vals = df_model_filtered[target_col].astype(str).str.lower().str.strip().unique()
            if df_model_filtered.empty: is_pure = True
            elif len(actual_vals) == 1 and actual_vals[0] == t_val: is_pure = True
        metrics["is_pure_target_class"] = is_pure

        if target_class_val is not None and target_col in df_model_filtered.columns:
            t_val = str(target_class_val).lower().strip()
            mask = df_model_filtered[target_col].astype(str).str.lower().str.strip() == t_val
            df_final_raw = df_model_filtered[mask]
        else: df_final_raw = df_model_filtered
            
        df_pred_norm = VectorizedUtils.normalize_dataframe(df_final_raw, valid_match_cols)
        df_gt_norm = VectorizedUtils.normalize_dataframe(df_gt_raw, valid_match_cols)
        set_res, _ = MetricCalculator.calc_set_metrics(df_pred_norm, df_gt_norm, valid_match_cols)
        metrics.update(set_res)
        # Classification 仍使用 F1 作为 final_score
        metrics["final_score"] = set_res["f1"]

    elif task_type == "regression":
        k = len(df_gt_raw)
        metrics["is_exact_topk_count"] = (len(df_model_filtered) == k)

        # ========== 改进点：使用健壮的 Top-K 选择 ==========
        if target_col not in df_model_filtered.columns:
            logger.warning(f"[{task_name}] 目标列 '{target_col}' 缺失，使用前 {k} 行")
            df_final_raw = df_model_filtered.head(k)
        else:
            # 转换为数值并过滤无效值
            target_series = pd.to_numeric(df_model_filtered[target_col], errors='coerce')
            valid_mask = target_series.notna()
            
            df_valid = df_model_filtered[valid_mask].copy()
            df_invalid = df_model_filtered[~valid_mask].copy()
            
            valid_count = len(df_valid)
            
            if valid_count == 0:
                # 没有有效值，使用原始前K行
                logger.warning(f"[{task_name}] 目标列无有效数值，使用前 {k} 行")
                df_final_raw = df_model_filtered.head(k)
            else:
                # 对有效值排序
                is_asc = (sort_dir == "low")
                sorted_series = target_series[valid_mask].sort_values(ascending=is_asc)
                top_k_indices = sorted_series.head(k).index
                
                df_topk_valid = df_model_filtered.loc[top_k_indices]
                
                # 如果有效值不足K个，补充无效值
                if len(df_topk_valid) < k and len(df_invalid) > 0:
                    remaining = k - len(df_topk_valid)
                    df_supplement = df_invalid.head(remaining)
                    df_final_raw = pd.concat([df_topk_valid, df_supplement], ignore_index=False)
                    logger.warning(
                        f"[{task_name}] 有效值仅 {valid_count} 个，"
                        f"补充 {len(df_supplement)} 个无效值"
                    )
                else:
                    df_final_raw = df_topk_valid
        # ========== 改进结束 ==========

        df_pred_norm = VectorizedUtils.normalize_dataframe(df_final_raw, valid_match_cols)
        df_gt_norm = VectorizedUtils.normalize_dataframe(df_gt_raw, valid_match_cols)
        set_res, match_pairs = MetricCalculator.calc_set_metrics(df_pred_norm, df_gt_norm, valid_match_cols)
        metrics.update(set_res)
        
        nmae = MetricCalculator.calc_nmae(df_final_raw, df_gt_raw, match_pairs, target_col, dataset_stats)
        metrics["nmae"] = nmae
        ndcg = MetricCalculator.calc_ndcg(df_final_raw, df_gt_raw, match_pairs, target_col, k, sort_direction=sort_dir)
        metrics["ndcg"] = ndcg
        
        # [新增统计] 计算 Kendall's Tau
        kendall = MetricCalculator.calc_kendall_tau(df_final_raw, df_gt_raw, match_pairs, target_col)
        metrics["kendall_tau"] = kendall

        # [修改] Regression 使用 Recall 作为主要指标计算 final_score
        # Recall 直接反映"找到了多少GT行"，是Regression任务的核心指标
        final = 0.4 * set_res["recall"] + 0.3 * ndcg + 0.3 * (1.0 - nmae)
        metrics["final_score"] = final

    return metrics

# ================= 主入口 =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_root", type=str, default="./outputs_run", help="Models output root")
    parser.add_argument("--dataset_root", type=str, default="./B4", help="Ground Truth dataset root")
    parser.add_argument("--models", nargs="+", default=["deepseek"], help="List of model names to evaluate")
    parser.add_argument("--mode", type=str, default="with_tool", help="Inference mode")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--disable-stats-cache", action="store_true", help="Use only existing stats_cache.json and never recompute stats from CSV")
    args = parser.parse_args()
    global DISABLE_STATS_CACHE
    DISABLE_STATS_CACHE = bool(args.disable_stats_cache)

    print("="*80)
    print(f"B4 EVALUATION ENGINE (Direct Match Mode)")
    print(f"Models: {args.models} | Mode: {args.mode}")
    print("="*80)

    for model in args.models:
        model_b4_root = os.path.join(args.inference_root, model, args.mode, "B4")
        if not os.path.exists(model_b4_root):
            logger.warning(f"Skip {model}: Path not found {model_b4_root}")
            continue

        tasks = []
        skipped_dataset_missing = [] 

        for root, dirs, files in os.walk(model_b4_root):
            has_output = any(f.endswith(".csv") and "current" in f for f in files)
            if has_output:
                rel_path = os.path.relpath(root, model_b4_root)
                ds_dir = os.path.join(args.dataset_root, rel_path)
                task_instance_name = os.path.basename(root)
                
                if os.path.exists(ds_dir):
                    tasks.append((root, ds_dir, task_instance_name))
                else:
                    skipped_dataset_missing.append(task_instance_name)
                    logger.warning(f"Found output but missing dataset dir: {ds_dir}")

        logger.info(f"Model {model}: Found {len(tasks)} tasks ready for eval.")
        
        results = []
        if args.workers <= 1:
            for m_dir, d_dir, t_name in tqdm(tasks, total=len(tasks), desc=f"Eval {model}"):
                try:
                    res = evaluate_single_task(m_dir, d_dir, t_name, model, args.mode)
                    res["task"] = t_name
                    results.append(res)
                except Exception as e:
                    logger.error(f"Task {t_name} CRITICAL FAIL: {e}")
                    results.append({"task": t_name, "error": f"CRITICAL_EXCEPTION: {str(e)}"})
        else:
            try:
                with ProcessPoolExecutor(max_workers=args.workers) as executor:
                    futures = {
                        executor.submit(evaluate_single_task, m_dir, d_dir, t_name, model, args.mode): t_name
                        for m_dir, d_dir, t_name in tasks
                    }
                    for f in tqdm(as_completed(futures), total=len(tasks), desc=f"Eval {model}"):
                        t_name = futures[f]
                        try:
                            res = f.result()
                            res["task"] = t_name
                            results.append(res)
                        except Exception as e:
                            logger.error(f"Task {t_name} CRITICAL FAIL: {e}")
                            results.append({"task": t_name, "error": f"CRITICAL_EXCEPTION: {str(e)}"})
            except PermissionError as e:
                logger.warning(f"Process pool unavailable ({e}); falling back to sequential evaluation.")
                for m_dir, d_dir, t_name in tqdm(tasks, total=len(tasks), desc=f"Eval {model}"):
                    try:
                        res = evaluate_single_task(m_dir, d_dir, t_name, model, args.mode)
                        res["task"] = t_name
                        results.append(res)
                    except Exception as inner_e:
                        logger.error(f"Task {t_name} CRITICAL FAIL: {inner_e}")
                        results.append({"task": t_name, "error": f"CRITICAL_EXCEPTION: {str(inner_e)}"})

        summary = {
            "model": model,
            "mode": args.mode,
            "classification": {"f1": [], "recall": [], "precision": [], "final": []},
            # [修改] Regression 增加独立的 recall 记录
            "regression": {"recall": [], "precision": [], "f1": [], "ndcg": [], "nmae": [], "kendall_tau": [], "final": []},
            "errors": [],
            "_stats_counters": {
                "total_tasks_detected": 0,    
                "total_tasks_successful": 0,  
                "not_empty_csv": 0,
                "csv_parse_error": 0, 
                "cols_match": 0,
                "extra_cols": 0, # 新增: 有多余列的任务数
                "missing_cols": 0, # 新增: 有缺失列的任务数
                "empty_result_filtered": 0, # 新增: 过滤后结果为空的任务数
                "filter_compliant": 0,
                "cls_pure_target": 0,
                "cls_zero_recall": 0, # 新增: 召回率为0的分类任务数
                "cls_total": 0,
                "reg_exact_topk": 0,
                "reg_total": 0
            }
        }
        
        detail_records = []
        for r in results:
            summary["_stats_counters"]["total_tasks_detected"] += 1
            
            if "error" in r:
                summary["errors"].append(f"{r['task']}: {r['error']}")
                continue 
            
            summary["_stats_counters"]["total_tasks_successful"] += 1

            if not r.get("is_csv_empty", True):
                summary["_stats_counters"]["not_empty_csv"] += 1
            if r.get("is_csv_parse_error", False):
                summary["_stats_counters"]["csv_parse_error"] += 1
            if r.get("cols_match_history", False):
                summary["_stats_counters"]["cols_match"] += 1
            if r.get("is_filter_compliant", False):
                summary["_stats_counters"]["filter_compliant"] += 1
            
            # [新增统计计数]
            if r.get("has_extra_cols", False):
                summary["_stats_counters"]["extra_cols"] += 1
            if r.get("has_missing_cols", False):
                summary["_stats_counters"]["missing_cols"] += 1
            if r.get("is_result_empty_after_filter", False):
                summary["_stats_counters"]["empty_result_filtered"] += 1

            # === 核心修复: 优先使用 result 中显式返回的 task_type ===
            t_type = r.get("task_type")
            
            if not t_type:
                if r.get("nmae", 0.0) != 0.0 or r.get("ndcg", 0.0) != 0.0:
                    t_type = "regression"
                else:
                    t_type = "classification"

            if t_type not in ["classification", "regression"]:
                logger.warning(f"Unknown task type '{t_type}' for {r['task']}, defaulting to classification for storage.")
                t_type = "classification"

            if t_type == "regression":
                # [修改] Regression 记录 recall 作为主指标
                summary[t_type]["recall"].append(r.get("recall", 0.0))
                summary[t_type]["precision"].append(r.get("precision", 0.0))
                summary[t_type]["f1"].append(r.get("f1", 0.0))  # F1 作为辅助
                summary[t_type]["ndcg"].append(r.get("ndcg", 0.0))
                summary[t_type]["nmae"].append(r.get("nmae", 0.0))
                summary[t_type]["kendall_tau"].append(r.get("kendall_tau", 0.0))
                summary["_stats_counters"]["reg_total"] += 1
                if r.get("is_exact_topk_count", False):
                    summary["_stats_counters"]["reg_exact_topk"] += 1
            else:
                summary[t_type]["recall"].append(r.get("recall", 0.0))
                summary[t_type]["precision"].append(r.get("precision", 0.0))
                summary[t_type]["f1"].append(r.get("f1", 0.0))
                summary["_stats_counters"]["cls_total"] += 1
                if r.get("is_pure_target_class", False):
                    summary["_stats_counters"]["cls_pure_target"] += 1
                if r.get("recall", 0.0) == 0.0: # 新增
                    summary["_stats_counters"]["cls_zero_recall"] += 1
            
            summary[t_type]["final"].append(r.get("final_score", 0.0))
            detail_records.append(r)

        final_report = {"model": model, "scores": {}, "statistics": {}}
        cnts = summary["_stats_counters"]
        total_detected = cnts["total_tasks_detected"] if cnts["total_tasks_detected"] > 0 else 1
        
        final_report["statistics"] = {
            "total_files_detected": total_detected,
            "successful_tasks": cnts["total_tasks_successful"],
            "failed_tasks": len(summary["errors"]),
            "csv_not_empty_rate": f"{cnts['not_empty_csv']/total_detected:.2%}",
            "csv_parse_error_rate": f"{cnts['csv_parse_error']/total_detected:.2%}",
            
            # 扩展详细分析指标
            "cols_match_history_rate": f"{cnts['cols_match']/total_detected:.2%}",
            "extra_cols_rate": f"{cnts['extra_cols']/total_detected:.2%}", # 新增
            "missing_cols_rate": f"{cnts['missing_cols']/total_detected:.2%}", # 新增
            "empty_result_after_filter_rate": f"{cnts['empty_result_filtered']/total_detected:.2%}", # 新增
            
            "filter_compliance_rate": f"{cnts['filter_compliant']/total_detected:.2%}",
            
            "classification_pure_target_rate": f"{cnts['cls_pure_target']/cnts['cls_total']:.2%}" if cnts['cls_total'] > 0 else "N/A",
            "classification_zero_recall_rate": f"{cnts['cls_zero_recall']/cnts['cls_total']:.2%}" if cnts['cls_total'] > 0 else "N/A", # 新增
            
            "regression_exact_topk_count_rate": f"{cnts['reg_exact_topk']/cnts['reg_total']:.2%}" if cnts['reg_total'] > 0 else "N/A"
        }

        for t_type in ["classification", "regression"]:
            s_dict = summary[t_type]
            if s_dict["final"]:
                stats_out = {
                    k: round(sum(v)/len(v), 4) for k, v in s_dict.items()
                }
                # [新增] 增加中位数统计，以对抗极端值干扰
                if "nmae" in s_dict and s_dict["nmae"]:
                    stats_out["nmae_median"] = round(float(np.median(s_dict["nmae"])), 4)
                if "f1" in s_dict and s_dict["f1"]:
                    stats_out["f1_median"] = round(float(np.median(s_dict["f1"])), 4)
                # [新增] Regression 增加 recall 中位数
                # if t_type == "regression" and "recall" in s_dict and s_dict["recall"]:
                #     stats_out["recall_median"] = round(float(np.median(s_dict["recall"])), 4)
                
                if "recall" in s_dict and s_dict["recall"]:
                    stats_out["recall_median"] = round(float(np.median(s_dict["recall"])), 4)

                if "precision" in s_dict and s_dict["precision"]:
                    stats_out["precision_median"] = round(float(np.median(s_dict["precision"])), 4)
                    
                final_report["scores"][t_type] = stats_out
                final_report["scores"][t_type]["count"] = len(s_dict["final"])
            else:
                final_report["scores"][t_type] = "No Data"
        
        if skipped_dataset_missing:
            final_report["skipped_dataset_missing"] = skipped_dataset_missing

        out_dir = os.path.join(args.inference_root, model, args.mode)
        os.makedirs(out_dir, exist_ok=True)
        safe_json_dump(detail_records, os.path.join(out_dir, "B4_details.json"))
        safe_json_dump(final_report, os.path.join(out_dir, "B4_summary.json"))
        
        print(f"\n[{model}] Evaluation Complete.")
        
        if summary["errors"]:
            print(f"⚠️  发现 {len(summary['errors'])} 个任务报错 (已从分数统计中剔除):")
            for err in summary["errors"]:
                print(f"   - {err}")
        
        if skipped_dataset_missing:
            print(f"⚠️  跳过 {len(skipped_dataset_missing)} 个任务 (数据集目录不存在):")
            for t in skipped_dataset_missing:
                print(f"   - {t}")
                
        print(json.dumps(final_report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

# python eval_b4_v3.py --models deepseek --mode aide_tool_gpt 
