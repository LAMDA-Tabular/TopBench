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
        """
        向量化清洗：替代原来的 clean_value 单点函数
        修复：强制将数字转为 float，防止布尔值减法报错
        """
        # 1. 统一转字符串并小写清洗 (作为默认值)
        str_cleaned = series.astype(str).str.strip().str.lower()
        
        # 2. 尝试转数字
        nums = pd.to_numeric(series, errors='coerce')
        mask_num = nums.notna() # 标记哪些是有效数字
        
        if not mask_num.any():
            return str_cleaned

        # === 核心修复 ===
        # 提取有效数字，并强制转为 float。
        valid_nums = nums[mask_num].astype(float)
        
        # 判断是否接近整数
        rounded = np.round(valid_nums)
        is_int = np.abs(valid_nums - rounded) < 1e-6
        
        # 格式化
        # Case A: 整数 (e.g. 5.0000001 -> "5")
        ints_formatted = rounded.astype(int).astype(str)
        
        # Case B: 浮点 (e.g. 5.123456 -> "5.1235")
        floats_formatted = np.round(valid_nums, 4).astype(str)
        
        # 合并数字结果
        final_nums = np.where(is_int, ints_formatted, floats_formatted)
        
        # 将处理好的数字填回原 Series
        result = str_cleaned.copy()
        result.loc[mask_num] = final_nums
        
        return result

    @staticmethod
    def normalize_dataframe(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """仅对目标列进行向量化清洗"""
        if df.empty: return df
        df_out = df.copy()
        
        # 清洗列名
        df_out.columns = [str(c).strip() for c in df_out.columns]
        
        valid_cols = [c for c in cols if c in df_out.columns]
        for col in valid_cols:
            df_out[col] = VectorizedUtils.clean_series_vectorized(df_out[col])
            
        return df_out

    # [Deleted] generate_hash_col has been removed as requested

# ================= 统计缓存管理器 =================
class StatsCache:
    """数据集统计缓存管理器 (ICML 审稿人建议：增强鲁棒性统计)"""
    
    @staticmethod
    def compute_and_cache_stats(base_dir: str) -> Dict:
        """计算并缓存数据集统计信息，包含用于 NMAE 计算的稳健分位数"""
        cache_path = os.path.join(base_dir, "stats_cache.json")
        
        # 尝试加载缓存
        # cached = safe_json_load(cache_path)
        # if cached:
        #     first_val = next(iter(cached.values()), {})
        #     if first_val.get("type") == "numeric" and "q95" in first_val:
        #         return cached
        #     logger.info("检测到旧版缓存或分位数缺失，正在重新计算统计数据...")
        
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
            
            if clean_series.empty:
                continue
            
            col_stat = {}
            
            # --- 核心修复：排除布尔型或强制转换 ---
            # pd.api.types.is_numeric_dtype(series) 对 bool 类型也返回 True
            # 我们通过 astype(float) 强制转换，确保减法和分位数计算安全
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                # 转换为 float 处理，防止 numpy 运算异常
                numeric_series = clean_series.astype(float)
                
                desc = numeric_series.describe()
                col_stat["type"] = "numeric"
                col_stat["std"] = float(desc.get("std", 0.0))
                col_stat["mean"] = float(desc.get("mean", 0.0))
                col_stat["min"] = float(desc.get("min", 0.0))
                col_stat["max"] = float(desc.get("max", 0.0))
                
                # 计算稳健分位数
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
            
            # --- 显式处理布尔型或类别型 ---
            else:
                unique_vals = clean_series.unique()
                n_unique = len(unique_vals)
                n_total = len(clean_series)
                
                # 如果是布尔型，我们将其视为特殊的类别型
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
        
        safe_json_dump(stats, cache_path)
        logger.info(f"统计缓存已成功写入: {cache_path}")
        return stats

# ================= 筛选引擎 =================
class FilterEngine:
    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: List[Dict]) -> Tuple[pd.DataFrame, float]:
        if not filters or df.empty:
            return df, 1.0

        temp_df = df.copy()
        for col in temp_df.columns:
            s_numeric = pd.to_numeric(temp_df[col], errors='coerce')
            mask_original_valid = temp_df[col].notna()
            mask_converted_nan = s_numeric.isna()
            loss_count = (mask_original_valid & mask_converted_nan).sum()

            if loss_count == 0:
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

# ================= 指标计算器 (修改核心) =================
class MetricCalculator:
    @staticmethod
    def calc_set_metrics(df_pred_norm: pd.DataFrame, df_gt_norm: pd.DataFrame, match_cols: List[str]) -> Tuple[Dict, List[Tuple[int, int]]]:
        if df_gt_norm.empty: return {"recall": 1.0, "precision": 1.0, "f1": 1.0}, []
        if df_pred_norm.empty: return {"recall": 0.0, "precision": 0.0, "f1": 0.0}, []

        # 重置索引以保留原始行号
        p_work = df_pred_norm.reset_index().rename(columns={'index': 'orig_idx'})
        g_work = df_gt_norm.reset_index().rename(columns={'index': 'orig_idx'})
        
        # -----------------------------------------------------------
        # 修改点：不再使用 Hash，而是基于列数据直接分组计算 cumcount
        # -----------------------------------------------------------
        
        # 为了防止 NaN 导致 merge 失败，填充一个特定的占位符进行分组
        # 注意：normalize_dataframe 已经把大多数数值转为字符串了，这里填充空串即可确保对齐
        p_filled = p_work[match_cols].fillna("<<NAN>>")
        g_filled = g_work[match_cols].fillna("<<NAN>>")

        # 计算累积计数，处理完全重复的行 (Handle duplicate rows)
        p_work['cc'] = p_filled.groupby(match_cols).cumcount()
        g_work['cc'] = g_filled.groupby(match_cols).cumcount()
        
        # 使用所有匹配列 + cc 进行合并
        merge_keys = match_cols + ['cc']
        
        # 执行合并
        # Pandas merge 默认不会匹配 NaN=NaN，但上面我们已经 fillna 处理了用于分组的 key，
        # 在 merge 时，如果原数据里有 NaN，为了匹配，我们需要确保 merge 的 key 也是无 NaN 的。
        # 最简单的做法是将临时的 filled 数据赋值回去，或者在 merge 时使用 filled 的列。
        # 这里为了不破坏原数据，我们把 filled 的列作为 key 加入到 dataframe 中用于 merge。
        
        p_work_merge = pd.concat([p_work, p_filled.add_suffix('_key')], axis=1)
        g_work_merge = pd.concat([g_work, g_filled.add_suffix('_key')], axis=1)
        
        merge_on_cols = [f"{c}_key" for c in match_cols] + ['cc']
        
        merged = pd.merge(
            p_work_merge, 
            g_work_merge, 
            on=merge_on_cols, 
            how='inner', 
            suffixes=('_p', '_g')
        )
        
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
    def calc_nmae(pred_df_raw: pd.DataFrame, gt_df_raw: pd.DataFrame, 
                  match_pairs: List[Tuple[int, int]], target_col: str,
                  dataset_stats: Dict) -> float:
        if gt_df_raw.empty: return 0.0
        
        t_stat = dataset_stats.get(target_col, {})
        # d_min = t_stat.get("min", 0)
        # d_max = t_stat.get("max", 0)
        # denom = (d_max - d_min) if (d_max - d_min) > 1e-5 else abs(t_stat.get("mean", 1.0))
        d_min = t_stat.get("q05", 0)
        d_max = t_stat.get("q95", 0)
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
        elif match_pairs:
            matched_error_sum = len(match_pairs) * 1.0
            
        count_missed = len(gt_df_raw) - len(match_pairs)
        total_error = matched_error_sum + (count_missed * 1.0)
        return min(total_error / len(gt_df_raw), 1.0)

    @staticmethod
    def calc_ndcg(pred_df_sorted: pd.DataFrame, gt_df_raw: pd.DataFrame, 
                  match_pairs: List[Tuple[int, int]], target_col: str, k: int,
                  sort_direction: str = 'high') -> float:
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

# ================= 单任务逻辑 =================
def evaluate_single_task(
    model_dir: str,       # 模型输出的具体文件夹
    dataset_dir: str,     # 数据集对应的具体文件夹
    task_name: str,       # 任务名 (文件夹名)
    model_name: str,      # 模型标识 (e.g. deepseek)
    mode_name: str        # 模式标识 (e.g. with_tool)
) -> Dict:
    
    # --- 1. 定位 Dataset 目录下的 Ground Truth 文件 ---
    gt_csv_path = os.path.join(dataset_dir, "test.csv")
    if not os.path.exists(gt_csv_path):
        gt_csv_path = os.path.join(dataset_dir, "test_current.csv")
    
    if not os.path.exists(gt_csv_path):
        return {"error": f"GT_CSV_MISSING: {gt_csv_path}"}

    # --- 2. 动态计算/加载 Dataset 目录下的 Stats Cache ---
    try:
        dataset_stats = StatsCache.compute_and_cache_stats(dataset_dir)
    except Exception as e:
        logger.warning(f"Stats计算失败 {dataset_dir}: {e}")
        dataset_stats = {}

    # --- 3. 定位 Model Output 目录下的文件 ---
    try:
        files = os.listdir(model_dir)
        
        # 查找 JSON (Metadata)
        json_candidates = [f for f in files if f.endswith(".json") and "current" in f and model_name in f]
        if not json_candidates:
            json_candidates = [f for f in files if f.endswith(".json") and "current" in f]
            
        json_filename = json_candidates[0] if json_candidates else None
        
        # 查找 CSV (Model Output)
        csv_candidates = [f for f in files if f.endswith(".csv") and "current" in f and model_name in f]
        if not csv_candidates:
             csv_candidates = [f for f in files if f.endswith(".csv") and "current" in f]
             
        csv_filename = csv_candidates[0] if csv_candidates else None
        
    except Exception as e:
        return {"error": f"FILE_SEARCH_ERROR: {e}"}

    if not json_filename: return {"error": "MODEL_JSON_MISSING"}
    if not csv_filename: return {"error": "MODEL_CSV_MISSING"}

    json_path = os.path.join(model_dir, json_filename)
    model_csv_path = os.path.join(model_dir, csv_filename)

    # --- 4. 读取 Metadata ---
    meta = safe_json_load(json_path)
    if not meta: return {"error": "META_JSON_INVALID"}
    
    gt_info = meta.get("ground_truth", {})
    task_type = gt_info.get("task_sub_type", "regression")
    target_col = gt_info.get("target_column")
    active_filters = gt_info.get("active_filters", [])
    target_class_val = gt_info.get("target_class_value") 
    sort_dir = gt_info.get("sort_direction", "high") 

    # --- 5. 数据加载与评估 ---
    try:
        df_model_raw = pd.read_csv(model_csv_path, on_bad_lines='skip', low_memory=False).reset_index(drop=True)
        df_gt_raw = pd.read_csv(gt_csv_path, on_bad_lines='skip', low_memory=False).reset_index(drop=True)
        
        df_model_raw.columns = [str(c).strip() for c in df_model_raw.columns]
        df_gt_raw.columns = [str(c).strip() for c in df_gt_raw.columns]
    except Exception as e:
        return {"error": f"CSV_READ_ERROR: {e}"}

    metrics = {
        "filter_compliance": 1.0,
        "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "nmae": 0.0, "ndcg": 0.0, "final_score": 0.0
    }

    match_cols = [c for c in df_gt_raw.columns if str(c).lower() != str(target_col).lower()]
    valid_match_cols = [c for c in match_cols if c in df_model_raw.columns]
    
    if not valid_match_cols:
        return {"error": "NO_MATCHING_COLS", "available": list(df_model_raw.columns)}

    # Step A: Filter
    if active_filters:
        df_model_filtered, compliance = FilterEngine.apply_filters(df_model_raw, active_filters)
        metrics["filter_compliance"] = compliance
    else:
        df_model_filtered = df_model_raw.copy()

    # Step B: Logic
    if task_type == "classification":
        if target_class_val is not None and target_col in df_model_filtered.columns:
            t_val = str(target_class_val).lower().strip()
            mask = df_model_filtered[target_col].astype(str).str.lower().str.strip() == t_val
            df_final_raw = df_model_filtered[mask]
        else:
            df_final_raw = df_model_filtered
            
        df_pred_norm = VectorizedUtils.normalize_dataframe(df_final_raw, valid_match_cols)
        df_gt_norm = VectorizedUtils.normalize_dataframe(df_gt_raw, valid_match_cols)
        
        set_res, _ = MetricCalculator.calc_set_metrics(df_pred_norm, df_gt_norm, valid_match_cols)
        metrics.update(set_res)
        metrics["final_score"] = set_res["f1"]

    elif task_type == "regression":
        k = len(df_gt_raw)
        if target_col in df_model_filtered.columns:
            s = pd.to_numeric(df_model_filtered[target_col], errors='coerce').fillna(-1e15)
            is_asc = (sort_dir == "low")
            top_k_indices = s.sort_values(ascending=is_asc).head(k).index
            df_final_raw = df_model_filtered.reindex(top_k_indices)
        else:
            df_final_raw = df_model_filtered.head(k)

        df_pred_norm = VectorizedUtils.normalize_dataframe(df_final_raw, valid_match_cols)
        df_gt_norm = VectorizedUtils.normalize_dataframe(df_gt_raw, valid_match_cols)

        set_res, match_pairs = MetricCalculator.calc_set_metrics(df_pred_norm, df_gt_norm, valid_match_cols)
        metrics.update(set_res)
        
        nmae = MetricCalculator.calc_nmae(df_final_raw, df_gt_raw, match_pairs, target_col, dataset_stats)
        metrics["nmae"] = nmae
        
        ndcg = MetricCalculator.calc_ndcg(
            df_final_raw, df_gt_raw, match_pairs, target_col, k, sort_direction=sort_dir
        )
        metrics["ndcg"] = ndcg
        
        final = 0.4 * set_res["f1"] + 0.3 * ndcg + 0.3 * (1.0 - nmae)
        metrics["final_score"] = final

    return metrics

# ================= 主入口 =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_root", type=str, default="./outputs", help="Models output root")
    parser.add_argument("--dataset_root", type=str, default="./B4", help="Ground Truth dataset root")
    parser.add_argument("--models", nargs="+", default=["deepseek"], help="List of model names to evaluate")
    parser.add_argument("--mode", type=str, default="with_tool", help="Inference mode")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    print("="*80)
    print(f"B4 EVALUATION ENGINE (Direct Match Mode)")
    print(f"Models: {args.models} | Mode: {args.mode}")
    print(f"Inference Root: {os.path.abspath(args.inference_root)}")
    print(f"Dataset Root:   {os.path.abspath(args.dataset_root)}")
    print("="*80)

    for model in args.models:
        model_b4_root = os.path.join(args.inference_root, model, args.mode, "B4")
        
        if not os.path.exists(model_b4_root):
            logger.warning(f"Skip {model}: Path not found {model_b4_root}")
            continue

        tasks = []
        for root, dirs, files in os.walk(model_b4_root):
            has_output = any(f.endswith(".csv") and "current" in f for f in files)
            
            if has_output:
                rel_path = os.path.relpath(root, model_b4_root)
                ds_dir = os.path.join(args.dataset_root, rel_path)
                task_instance_name = os.path.basename(root)
                
                if os.path.exists(ds_dir):
                    tasks.append((root, ds_dir, task_instance_name))
                else:
                    logger.warning(f"Found output but missing dataset dir: {ds_dir}")

        logger.info(f"Model {model}: Found {len(tasks)} valid tasks.")
        
        results = []
        if args.workers <= 1:
            for m_dir, d_dir, t_name in tqdm(tasks, total=len(tasks), desc=f"Eval {model}"):
                try:
                    res = evaluate_single_task(m_dir, d_dir, t_name, model, args.mode)
                    res["task"] = t_name
                    results.append(res)
                except Exception as e:
                    logger.error(f"Task {t_name} CRITICAL FAIL: {e}")
                    logger.error(traceback.format_exc())
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
                            logger.error(traceback.format_exc())
            except PermissionError as e:
                logger.warning(f"Process pool unavailable ({e}); falling back to sequential evaluation.")
                for m_dir, d_dir, t_name in tqdm(tasks, total=len(tasks), desc=f"Eval {model}"):
                    try:
                        res = evaluate_single_task(m_dir, d_dir, t_name, model, args.mode)
                        res["task"] = t_name
                        results.append(res)
                    except Exception as inner_e:
                        logger.error(f"Task {t_name} CRITICAL FAIL: {inner_e}")
                        logger.error(traceback.format_exc())

        summary = {
            "model": model,
            "mode": args.mode,
            "classification": {"f1": [], "recall": [], "precision": [], "final": []},
            "regression": {"f1": [], "ndcg": [], "nmae": [], "final": []},
            "errors": []
        }
        
        detail_records = []
        for r in results:
            if "error" in r:
                summary["errors"].append(f"{r['task']}: {r['error']}")
                continue
            
            if "nmae" in r and (r["nmae"] != 0.0 or r["ndcg"] != 0.0):
                t_type = "regression"
                summary[t_type]["ndcg"].append(r["ndcg"])
                summary[t_type]["nmae"].append(r["nmae"])
            else:
                t_type = "classification"
                summary[t_type]["recall"].append(r["recall"])
                summary[t_type]["precision"].append(r["precision"])
            
            summary[t_type]["f1"].append(r["f1"])
            summary[t_type]["final"].append(r["final_score"])
            detail_records.append(r)

        final_report = {"model": model, "scores": {}}
        for t_type in ["classification", "regression"]:
            s_dict = summary[t_type]
            if s_dict["final"]:
                final_report["scores"][t_type] = {
                    k: round(sum(v)/len(v), 4) for k, v in s_dict.items()
                }
                final_report["scores"][t_type]["count"] = len(s_dict["final"])
            else:
                final_report["scores"][t_type] = "No Data"

        out_dir = os.path.join(args.inference_root, model, args.mode)
        os.makedirs(out_dir, exist_ok=True)
        safe_json_dump(detail_records, os.path.join(out_dir, "B4_details.json"))
        safe_json_dump(final_report, os.path.join(out_dir, "B4_summary.json"))
        
        print(f"\n[{model}] Evaluation Complete.")
        print(json.dumps(final_report, indent=2))

if __name__ == "__main__":
    main()
