import os
import json
import pandas as pd
import numpy as np
import threading
import time
import itertools
import re
import traceback
import difflib
import math
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI, APIStatusError, RateLimitError, AuthenticationError, APIConnectionError
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
import argparse
import asyncio
from openai import AsyncOpenAI
from pathlib import Path

try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_JUDGE_AVAILABLE = True
except Exception:
    genai = None
    genai_types = None
    GEMINI_JUDGE_AVAILABLE = False

# ================= 日志配置 =================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================= 配置部分 =================


def _env_keys(name: str) -> List[str]:
    return [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]


LEGACY_DATA_ROOT = Path(os.getenv("TOPBENCH_LEGACY_DATA_ROOT", Path.cwd())).resolve()


@dataclass
class ScoringConfig:
    """评分配置"""
    # --- B1 (单点) 权重 ---
    accuracy_weight: float = 0.6
    logic_weight: float = 0.4
    
    # --- B2 (多方案) 权重 (新增) ---
    # 建议: 决策(0.4) + 平均预测精度(0.2) + 逻辑(0.4) = 1.0
    # 或者: 决策(0.3) + 平均预测精度(0.3) + 逻辑(0.4) = 1.0
    b2_decision_weight: float = 0.5
    b2_avg_pred_weight: float = 0.3
    b2_logic_weight: float = 0.2

    b3_trend_weight: float = 0.5
    b3_pred_002_weight: float = 0.3
    b3_logic_weight: float = 0.2

    numeric_tolerance: float = 0.05
    string_similarity_threshold: float = 0.85

    # 区间惩罚配置
    # 阈值：预测宽度是真实宽度的多少倍开始惩罚 (例如 2.0 倍)
    overlap_penalty_width_factor: float = 2.0 
    # 指数衰减系数 alpha。值越大，超出阈值后分数下降越快。
    # 例如 alpha=1.0, 超过阈值1倍时，得分乘 e^-1 (约0.36)
    width_penalty_alpha: float = 1.0
    
    def validate(self):
        """验证配置合法性"""
        if not math.isclose(self.accuracy_weight + self.logic_weight, 1.0):
            raise ValueError("B1 权重之和必须为1.0")
        # 验证 B2 权重
        if not math.isclose(self.b2_decision_weight + self.b2_avg_pred_weight + self.b2_logic_weight, 1.0):
            raise ValueError("B2 权重之和必须为1.0")
        
        if not math.isclose(self.b3_trend_weight + self.b3_pred_002_weight + self.b3_logic_weight, 1.0):
            raise ValueError("B3 权重之和必须为1.0")

        if self.numeric_tolerance < 0 or self.numeric_tolerance > 1:
            raise ValueError("numeric_tolerance必须在[0, 1]范围内")

@dataclass
class PathConfig:
    """路径配置"""
    # 推理结果根目录 (保持不变)
    inference_root: str = "outputs"
    
    # 目标任务集 ("B1" 或 "B2")
    target_benchmark: str = "B1" 
    
    # 模型和模式过滤
    target_models: Optional[List[str]] = None
    target_mode: Optional[str] = None # "no_tool" / "with_tool"

    @property
    def dataset_root(self) -> str:
        """根据 benchmark 动态返回数据集路径"""
        if self.target_benchmark == "B1":
            return str(LEGACY_DATA_ROOT / "B1andB3" / "B1")
        elif self.target_benchmark == "B2":
            return str(LEGACY_DATA_ROOT / "B2")
        elif self.target_benchmark == "B3":
            # [新增] B3 路径
            return str(LEGACY_DATA_ROOT / "B3")
        else:
            raise ValueError(f"未知的 benchmark: {self.target_benchmark}")

    def validate(self):
        if not os.path.exists(self.dataset_root):
            raise ValueError(f"数据集根目录不存在: {self.dataset_root}")
        if not os.path.exists(self.inference_root):
            raise ValueError(f"推理结果根目录不存在: {self.inference_root}")
        
# ================= 配置部分 =================

SCORING_CONFIG = ScoringConfig()
SCORING_CONFIG.validate()

def parse_args():
    parser = argparse.ArgumentParser(description="Table Prediction Evaluation Script")
    
    # 推理根目录
    parser.add_argument("--inference_root", type=str, 
                        default="outputs",
                        help="Path to inference results root directory")
    
    # Benchmark 类型
    parser.add_argument("--benchmark", type=str, default="B2", 
                        choices=["B1", "B2", "B3"],
                        help="Target benchmark (B1, B2, or B3)")
    
    # 目标模型列表 (支持传入多个，例如: --models qwen_thinking deepseek)
    parser.add_argument("--models", nargs="+", default=["qwen_thinking"],
                        help="Specific models to evaluate (space separated)")
    
    # 模式选择
    parser.add_argument("--mode", type=str, default="no_tool",
                        choices=["no_tool", "with_tool", "aide_tool_gpt"],
                        help="Target mode (no_tool or with_tool)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel evaluation workers")
    parser.add_argument("--disable-stats-cache", action="store_true",
                        help="Use only existing stats_cache.json during evaluation and never recompute stats from CSV")
    parser.add_argument("--export-scale-breakdown", action="store_true",
                        help="Export additional summaries grouped by table scale buckets")
    parser.add_argument("--export-shape-breakdown", action="store_true",
                        help="Export additional summaries grouped by row-scale and column-count buckets")
    parser.add_argument("--export-metric-sensitivity", action="store_true",
                        help="Export additional summaries under alternative metric hyperparameter settings without changing the default evaluation results")
    parser.add_argument("--export-metric-sensitivity-full", action="store_true",
                        help="Export a fuller metric hyperparameter grid and ranking-stability analyses without changing the default evaluation results")
    parser.add_argument("--export-metric-rank-stability", action="store_true",
                        help="Export ranking-stability analyses across metric hyperparameter settings")
    parser.add_argument("--analysis-use-all-inference-files", action="store_true",
                        help="Build analysis exports from all matching inference files, even when existing eval files are skipped")
    parser.add_argument("--dump-judge-artifacts", action="store_true",
                        help="Dump judge model metadata and prompt templates for reproducibility")
    parser.add_argument("--judge-backend", type=str, default="deepseek", choices=["deepseek", "gpt", "gemini"],
                        help="Judge backend for extraction and logic scoring")
    parser.add_argument("--judge-model-id", type=str, default=None,
                        help="Optional override for the judge model identifier")
    parser.add_argument("--eval-suffix", type=str, default="_eval",
                        help="Suffix for saved evaluation files, e.g. _eval or _eval_gemini_judge")
    parser.add_argument("--no-skip-existing-eval", action="store_true",
                        help="Re-evaluate files even if the target eval JSON already exists")
    parser.add_argument("--scale-thresholds", nargs=2, type=int, default=[1000, 100000],
                        metavar=("SHORT_MAX", "MEDIUM_MAX"),
                        help="Row-count thresholds for Short/Medium/Long scale buckets")
    parser.add_argument("--metric-weight-grid", nargs="+", type=float, default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                        help="Internal point-score weight grid used for accuracy metric sensitivity (interval weight = 1 - point weight)")
    parser.add_argument("--metric-width-factors", nargs="+", type=float, default=[1.5, 2.0, 3.0],
                        help="Width-factor grid used for interval penalty sensitivity analyses")
    parser.add_argument("--metric-width-alphas", nargs="+", type=float, default=[0.5, 1.0, 2.0],
                        help="Alpha grid used for interval penalty sensitivity analyses")

    return parser.parse_args()

# [修改] 初始化 PATH_CONFIG 并应用命令行参数
args = parse_args()

PATH_CONFIG = PathConfig()
PATH_CONFIG.inference_root = args.inference_root
PATH_CONFIG.target_benchmark = args.benchmark
PATH_CONFIG.target_models = args.models
PATH_CONFIG.target_mode = args.mode

# 打印配置日志以确认
logger.info(f"Config Applied: Benchmark={PATH_CONFIG.target_benchmark}, "
            f"Mode={PATH_CONFIG.target_mode}, Models={PATH_CONFIG.target_models}")

PATH_CONFIG.validate()

# PATH_CONFIG = PathConfig()
# # 可以指定特定模型，例如：PATH_CONFIG.target_models = ["deepseek", "gpt-4"]
# PATH_CONFIG.target_models = ["qwen_thinking"]
# PATH_CONFIG.target_mode = "no_tool"
# PATH_CONFIG.validate()

JUDGE_BACKEND = args.judge_backend
if JUDGE_BACKEND == "gemini":
    if not GEMINI_JUDGE_AVAILABLE:
        raise ImportError("google-genai is required for --judge-backend gemini")
    API_KEYS = _env_keys("GEMINI_API_KEY")
    BASE_URL = None
    default_judge_model_id = os.getenv("GEMINI_JUDGE_MODEL_ID", "gemini-3-flash-preview")
elif JUDGE_BACKEND == "gpt":
    API_KEYS = _env_keys("OPENAI_API_KEY")
    BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    default_judge_model_id = os.getenv("OPENAI_JUDGE_MODEL_ID", "gpt-5.2")
else:
    API_KEYS = _env_keys("DEEPSEEK_API_KEY")
    BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    default_judge_model_id = os.getenv("DEEPSEEK_JUDGE_MODEL_ID", "deepseek-chat")

if not API_KEYS:
    raise ValueError(f"No API key configured for judge backend '{JUDGE_BACKEND}'.")

key_iterator = itertools.cycle(API_KEYS)
JUDGE_MODEL_ID = args.judge_model_id or default_judge_model_id
MAX_WORKERS = max(1, int(args.workers))
DISABLE_STATS_CACHE = bool(args.disable_stats_cache)
EXPORT_SCALE_BREAKDOWN = bool(args.export_scale_breakdown)
EXPORT_SHAPE_BREAKDOWN = bool(args.export_shape_breakdown)
EXPORT_METRIC_SENSITIVITY = bool(args.export_metric_sensitivity)
EXPORT_METRIC_SENSITIVITY_FULL = bool(args.export_metric_sensitivity_full)
EXPORT_METRIC_RANK_STABILITY = bool(args.export_metric_rank_stability)
ANALYSIS_USE_ALL_INFERENCE_FILES = bool(args.analysis_use_all_inference_files)
DUMP_JUDGE_ARTIFACTS = bool(args.dump_judge_artifacts)
SCALE_THRESHOLDS = tuple(sorted(int(x) for x in args.scale_thresholds))
EVAL_SUFFIX = str(args.eval_suffix).strip() or "_eval"
SKIP_EXISTING_EVAL = not bool(args.no_skip_existing_eval)
METRIC_WEIGHT_GRID = sorted(set(round(float(x), 4) for x in args.metric_weight_grid if 0.0 < float(x) < 1.0))
METRIC_WIDTH_FACTORS = sorted(set(round(float(x), 4) for x in args.metric_width_factors if float(x) > 0))
METRIC_WIDTH_ALPHAS = sorted(set(round(float(x), 4) for x in args.metric_width_alphas if float(x) > 0))

if len(SCALE_THRESHOLDS) != 2:
    raise ValueError("--scale-thresholds must provide exactly two integers")
if not METRIC_WEIGHT_GRID:
    raise ValueError("--metric-weight-grid must contain values strictly between 0 and 1")
if not METRIC_WIDTH_FACTORS:
    raise ValueError("--metric-width-factors must contain positive values")
if not METRIC_WIDTH_ALPHAS:
    raise ValueError("--metric-width-alphas must contain positive values")


def get_eval_path(inference_path: str) -> str:
    return inference_path.replace(".json", f"{EVAL_SUFFIX}.json")


class _SimpleMessage:
    def __init__(self, content: str):
        self.content = content


class _SimpleChoice:
    def __init__(self, content: str):
        self.message = _SimpleMessage(content)


class _SimpleCompletion:
    def __init__(self, content: str):
        self.choices = [_SimpleChoice(content)]

# ================= 路径管理 =================
class PathManager:
    """路径管理器，处理推理文件和数据集文件的映射"""
    
    @staticmethod
    def get_dataset_path_from_inference(inference_path: str, 
                                    inference_root: str, 
                                    dataset_root: str) -> str:
        """
        根据推理文件路径获取对应的数据集路径
        兼容 B1 (扁平) 和 B2 (含 regression/classification) 结构
        """
        inference_dir = os.path.dirname(inference_path)
        dataset_name = os.path.basename(inference_dir)
        
        # --- 策略 1: 解析相对路径进行精确查找 ---
        try:
            parts = inference_dir.split(os.sep)
            inference_root_parts = inference_root.split(os.sep)
            
            # 找到 benchmark (B1/B2) 之后的部分
            # 假设推理路径结构: .../B1/domain/dataset_name
            # 提取出 ['domain', 'dataset_name']
            relative_parts = []
            found_benchmark = False
            skip_count = 0 # 跳过 model 和 mode
            
            # 计算 inference_root 的深度，以便从正确位置开始切片
            root_depth = len([p for p in inference_root_parts if p])
            
            for part in parts[root_depth:]:
                # 跳过 model_name 和 mode 两层
                if skip_count < 2:
                    skip_count += 1
                    continue
                
                if found_benchmark:
                    relative_parts.append(part)
                elif part in ['B1', 'B2', 'B3']:
                    found_benchmark = True
            
            if relative_parts:
                # 1.1 尝试 B1 风格的直接拼接
                # Path: root/domain/dataset_name
                candidate_direct = os.path.join(dataset_root, *relative_parts)
                if os.path.exists(os.path.join(candidate_direct, "info.json")):
                    return candidate_direct
                
                # 1.2 尝试 B2 风格的插入拼接 (尝试插入 regression 或 classification)
                # Path: root/domain/{task_type}/dataset_name
                # 假设 relative_parts 是 ['finance', 'Bitcoin_B2']
                if len(relative_parts) >= 2:
                    domain = relative_parts[0]
                    ds_name = relative_parts[-1]
                    
                    for sub_type in ["regression", "classification"]:
                        candidate_nested = os.path.join(dataset_root, domain, sub_type, ds_name)
                        if os.path.exists(os.path.join(candidate_nested, "info.json")):
                            return candidate_nested

        except Exception as e:
            logger.warning(f"路径解析策略失败: {e}，尝试暴力搜索")

        # --- 策略 2: 暴力搜索 (兜底) ---
        # 如果目录结构极其不规则，直接在 dataset_root 下递归寻找名为 dataset_name 的文件夹
        # 且该文件夹内必须包含 info.json
        for root, dirs, files in os.walk(dataset_root):
            if dataset_name in dirs:
                candidate_walk = os.path.join(root, dataset_name)
                if os.path.exists(os.path.join(candidate_walk, "info.json")):
                    # logger.info(f"通过搜索找到数据集: {candidate_walk}")
                    return candidate_walk
        
        # 如果都找不到，返回推理目录(虽然基本没用，但防止崩溃)
        logger.warning(f"未找到数据集路径: {dataset_name}")
        return inference_dir

    # ... 其他方法 (extract_mode_from_path, extract_model_name, collect_inference_files) 保持不变 ...
    @staticmethod
    def extract_mode_from_path(inference_path: str, inference_root: str) -> str:
        """从推理文件路径提取 mode"""
        try:
            rel_path = os.path.relpath(inference_path, inference_root)
            parts = rel_path.split(os.sep)
            if len(parts) >= 2:
                return parts[1]  # model/mode/... 中的 mode
            return "unknown"
        except Exception as e:
            logger.warning(f"提取mode失败: {e}")
            return "unknown"
    
    @staticmethod
    def extract_model_name(inference_path: str, inference_root: str) -> str:
        """从推理文件路径提取模型名称"""
        try:
            rel_path = os.path.relpath(inference_path, inference_root)
            model_name = rel_path.split(os.sep)[0]
            return model_name
        except Exception as e:
            logger.warning(f"提取模型名称失败: {e}")
            return "unknown"
    
    @staticmethod
    def collect_inference_files(inference_root: str, 
                            target_models: Optional[List[str]] = None,
                            target_benchmark: str = "B1",
                            target_mode: Optional[str] = None,  # 新增参数
                            judge_model_id: str = "deepseek-chat",
                            skip_existing_eval: Optional[bool] = None) -> List[Tuple[str, str, str]]:  # 返回增加 mode
        """
        收集需要评估的推理文件
        返回: List[(inference_file_path, model_name, mode)]
        """
        tasks = []
        skipped_existing_eval = 0
        use_skip_existing_eval = SKIP_EXISTING_EVAL if skip_existing_eval is None else bool(skip_existing_eval)
        
        if not os.path.exists(inference_root):
            logger.error(f"推理根目录不存在: {inference_root}")
            return tasks
        
        # 遍历模型目录
        for model_name in os.listdir(inference_root):
            model_path = os.path.join(inference_root, model_name)
            
            if not os.path.isdir(model_path):
                continue
            
            # 如果指定了目标模型，检查是否匹配
            if target_models and model_name not in target_models:
                # logger.info(f"跳过模型: {model_name}")
                continue
            
            # 遍历 mode 目录
            for mode in os.listdir(model_path):
                mode_path = os.path.join(model_path, mode)
                
                if not os.path.isdir(mode_path):
                    continue
                
                # 如果指定了目标 mode，检查是否匹配
                if target_mode and mode != target_mode:
                    # logger.info(f"跳过模式: {model_name}/{mode}")
                    continue
                
                # 查找 benchmark 目录
                benchmark_path = os.path.join(mode_path, target_benchmark)
                if not os.path.exists(benchmark_path):
                    # logger.warning(f"模型 {model_name}/{mode} 下未找到 {target_benchmark} 目录")
                    continue
                
                # 递归查找JSON文件
                for root, dirs, files in os.walk(benchmark_path):
                    for file in files:
                        if (file.endswith(".json") and 
                            "tool" in file and
                            "_eval" not in file):
                            file_path = os.path.join(root, file)
                            if use_skip_existing_eval and os.path.exists(get_eval_path(file_path)):
                                skipped_existing_eval += 1
                                continue
                            tasks.append((file_path, model_name, mode))  # 增加 mode

        if use_skip_existing_eval and skipped_existing_eval:
            logger.info(f"检测到已有评测结果，跳过 {skipped_existing_eval} 个已完成文件 ({EVAL_SUFFIX})")
        return tasks

# ================= 工具函数 =================
class NumpyEncoder(json.JSONEncoder):
    """处理numpy类型的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)

def safe_json_dump(data: Any, filepath: str, **kwargs) -> bool:
    """安全的JSON写入"""
    try:
        # 修改点：如果外部没有传入 indent，则默认设置为 2
        kwargs.setdefault('indent', 2)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # 修改点：从参数列表中移除硬编码的 indent=2，完全由 kwargs 控制
            json.dump(data, f, cls=NumpyEncoder, ensure_ascii=False, **kwargs)
        return True
    except Exception as e:
        logger.error(f"写入JSON失败 {filepath}: {e}")
        return False

def safe_json_load(filepath: str) -> Optional[Dict]:
    """安全的JSON加载"""
    if not os.path.exists(filepath):
        logger.warning(f"文件不存在: {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"读取JSON失败 {filepath}: {e}")
        return None

def extract_json_content(text: str) -> str:
    """从文本中提取JSON内容"""
    if not text:
        return ""
    
    text = text.strip()
    
    # 移除Markdown代码块
    if "```json" in text:
        parts = re.split(r"```json", text)
        if len(parts) > 1:
            text = re.split(r"```", parts[1])[0]
    elif "```" in text:
        parts = re.split(r"```", text)
        if len(parts) > 1:
            text = parts[1]
    
    # 提取第一个完整的JSON对象
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return text[start_idx:end_idx + 1]
    
    return text

def normalize_text(s: Any) -> str:
    """标准化文本，用于比较 (修改版：解决 0.0 != 0 的问题)"""
    if pd.isna(s) or s is None:
        return ""
    
    s = str(s).strip().lower()
    # 移除多余的空白字符
    s = re.sub(r'\s+', ' ', s)
    # 移除常见标点
    s = s.rstrip('.')
    
    # [新增] 尝试数值归一化
    # 目的：使 "0.0", "0.00", "0" 都能统一转为 "0"
    # 同时使 "1.50" 统一转为 "1.5"
    try:
        f_val = float(s)
        # 如果是整数 (例如 0.0, 1.00, 25.0)
        if f_val.is_integer():
            return str(int(f_val))
        # 如果是浮点数 (例如 0.50 -> 0.5)
        return str(f_val)
    except (ValueError, TypeError):
        # 转换失败说明是纯文本 (例如 "cat")，保持原样
        pass
    
    return s

def calculate_string_similarity(pred: str, gt: str) -> float:
    """计算字符串相似度"""
    norm_pred = normalize_text(pred)
    norm_gt = normalize_text(gt)
    
    if not norm_gt:
        return 0.0
    if norm_pred == norm_gt:
        return 1.0
    
    return difflib.SequenceMatcher(None, norm_pred, norm_gt).ratio()

# ================= 数值解析与幻觉检测 =================
class MagnitudeChecker:
    """量级检查器"""
    
    @staticmethod
    def check_magnitude_and_bounds(val: Any, target_stat: Dict, response: str) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        检查预测值是否严重偏离数据分布
        [修改]: 增加返回值 corrected_val，用于返回经符号修正后的数值
        """
        if val is None or not target_stat:
            return True, None, val
        
        min_v = target_stat.get("min")
        max_v = target_stat.get("max")
        
        if min_v is None or max_v is None:
            return True, None, val
        
        try:
            val_f = float(val)
            
            # [核心修改] 负数强制转正逻辑
            # 如果目标范围全是正数 (min >= 0)，但预测值是负数，则强行取绝对值
            if min_v >= 0 and val_f < 0:
                val_f = abs(val_f)
            
            data_range = max_v - min_v
            
            # 处理极小范围的情况
            if data_range < 1e-6:
                data_range = max(abs(max_v) * 0.1, 1e-6)
            
            # 定义宽松边界
            lower_bound = min_v - 5.0 * data_range
            upper_bound = max_v + 5.0 * data_range
            
            # 范围检查 (使用可能修正后的 val_f)
            if val_f < lower_bound or val_f > upper_bound:
                return False, (
                    f"The predicted_value {val_f} is inconsistent with the target magnitude."
                    f"The effective target range is approximately [{min_v:.2f}, {max_v:.2f}]."
                    f"Please check if any unit suffixes (k, m, %) are missing or if there are any errors in the magnitude."
                    f"Please check [Model Response] again: '{response}'"
                ), val_f
            
            return True, None, val_f
            
        except (ValueError, TypeError) as e:
            logger.warning(f"量级检查转换失败: {e}")
            return True, None, val

import math
import re
from typing import Set, Any, Optional

# 尝试导入 word2number，如果没有安装则降级运行
try:
    from word2number import w2n
except ImportError:
    w2n = None
    print("Warning: 'word2number' library not found. Please install via `pip install word2number` for better text parsing.")

class NumberParser:
    """数值解析器 (修复版：修复单位后缀后紧跟连字符被误判为负号的问题)"""
    
    MULTIPLIERS = {
        'k': 1e3,
        'm': 1e6, 'mn': 1e6, 'million': 1e6,
        'b': 1e9, 'bn': 1e9, 'billion': 1e9,
        '%': 0.01,
        'lakh': 1e5, 'lakhs': 1e5, 'lac': 1e5, 'lacs': 1e5,
        'cr': 1e7, 'crore': 1e7, 'crores': 1e7, 'lpa': 1e5, 'cpa': 1e7
    }

    ZERO_EQUIVALENTS = {
        "no one", "nobody", "none", "nothing", "nil", 
        "zero", "no person", "no people", "neither", "no"
    }

    @staticmethod
    def parse_text_number_to_values(text: str) -> Set[float]:
        """从文本中解析所有可能的数值 (修复版：优化负号/区间判定逻辑)"""
        if not text:
            return set()
        
        found_values = set()
        clean_text = text.lower().strip()
        
        # 1. 优先处理 Unicode 减号和不规范的负号
        clean_text = clean_text.replace('−', '-').replace('–', '-').replace('—', '-')
        
        # 2. 语义零值检测
        for zero_word in NumberParser.ZERO_EQUIVALENTS:
            pattern = r'\b' + re.escape(zero_word) + r'\b'
            if re.search(pattern, clean_text):
                found_values.add(0.0)
                found_values.add(0)

        # 3. Word2Number (尝试解析英文数字单词)
        if 'w2n' in globals() and w2n:
            text_for_w2n = re.sub(r'(?<=\w)\s*[-–—]\s*(?=\w)', ' ', clean_text).replace(',', '')
            try:
                val = w2n.word_to_num(text_for_w2n)
                found_values.add(float(val))
            except ValueError: pass
            
            words = re.split(r'[\s,]+', text_for_w2n)
            for word in words:
                word_clean = word.strip(".,!?;:\"'()[]{}*")
                if not word_clean: continue
                try:
                    val = w2n.word_to_num(word_clean)
                    found_values.add(float(val))
                except ValueError: continue

        # --- 正则表达式提取准备 ---
        clean_text_for_regex = clean_text
        
        # 处理 LaTeX 格式的千分位
        clean_text_for_regex = clean_text_for_regex.replace('{,}', ',')
        
        # 移除干扰符号 (货币、大约、大于小于等)，保留正负号
        symbols_to_remove = ['$', '¥', '€', '£', '￥', '₹', 'rp', 'rs', '~', '>', '<', 'approx', '≈', '≥', '≤']
        for symbol in symbols_to_remove:
            clean_text_for_regex = clean_text_for_regex.replace(symbol, '')
        
        suffix_str = r'(?:k|m|mn|b|bn|billion|million|lakhs?|lacs?|lpa|crores?|cpa|%)'
        
        # ================= [区间单位共用处理] =================
        # 处理 "10-20k" 这种共享后缀的情况
        range_shared_suffix_pattern = r'(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*(' + suffix_str + r')'
        range_matches = re.findall(range_shared_suffix_pattern, clean_text_for_regex)
        
        for n1_str, n2_str, suffix in range_matches:
            try:
                clean_suffix = suffix.strip()
                mult = NumberParser.MULTIPLIERS.get(clean_suffix, 1.0)
                v1 = float(n1_str) * mult
                v2 = float(n2_str) * mult
                found_values.add(v1)
                found_values.add(v2)
            except ValueError: continue
        # ==============================================================

        # 4. 常规正则提取
        
        sanitized_text = clean_text_for_regex
        
        # A. 替换无歧义分隔符
        sanitized_text = re.sub(r'/| to ', ' ', sanitized_text)
        
        # B. [关键修复] 智能替换区间连字符 (区分负号和区间)
        # 旧逻辑: (?<=[0-9%]) 只允许数字或%后的 "-" 变为空格，导致 "140k-160k" 中的 "-" 被保留，变成负号
        # 新逻辑: (?<=[0-9%a-zA-Z]) 允许字母后缀后的 "-" 也变为空格。
        # 原理: 真正的负号通常前面是空格或行首 (如 "Change: -10")，而不会紧跟在字母后面 (如 "Year-2020", "Zone-A", "140k-160k")
        sanitized_text = re.sub(r'(?<=[0-9%a-zA-Z])\s*-\s*(?=[0-9])', ' ', sanitized_text)
        
        # 匹配 "可选正负号 + 数字 + 后缀"
        pattern = r'([\+\-]?\d+(?:,\d+)*(?:\.\d+)?)\s*(' + suffix_str + r')?'
        matches = re.findall(pattern, sanitized_text)
        
        for num_str, suffix in matches:
            if not num_str: continue
            try:
                # 移除逗号
                raw_val_str = num_str.replace(',', '')
                if raw_val_str in ['-', '+', '.']: continue 
                
                raw_val = float(raw_val_str)
                found_values.add(raw_val) 
                
                if suffix:
                    clean_suffix = suffix.strip()
                    if clean_suffix in NumberParser.MULTIPLIERS:
                        mult = NumberParser.MULTIPLIERS[clean_suffix]
                        found_values.add(raw_val * mult)
                        if clean_suffix == '%':
                            found_values.add(raw_val)
            except ValueError: continue
            
        return found_values

    @staticmethod
    def is_value_match(target: Any, text_values_set: Set[float], tolerance: float = 0.01) -> bool:
        """检查目标值是否在文本值集合中"""
        if target is None:
            return True
        
        try:
            if isinstance(target, str):
                target = target.replace(',', '')
            target_float = float(target)
        except (ValueError, TypeError):
            return False
        
        ALLOWED_SCALES = {0.01, 100.0, 1000.0, 1e6, 1e9}
        
        for v in text_values_set:
            # 1. 绝对/相对误差匹配
            if math.isclose(v, target_float, rel_tol=tolerance, abs_tol=1e-5):
                return True
            
            # 2. 倍率匹配 (防止单位未对齐，如 150 vs 150k，允许一定倍率偏差)
            try:
                if abs(v) > 1e-9:
                    ratio = target_float / v
                    for scale in ALLOWED_SCALES:
                        if (math.isclose(ratio, scale, rel_tol=tolerance) or 
                            math.isclose(ratio, 1.0/scale, rel_tol=tolerance)):
                            return True
            except Exception: pass
        
        return False

class HallucinationVerifier:
    """幻觉检测器 (增强版：支持拼接引用匹配 + 数字锚点宽松匹配 + ±区间推导 + 模糊Token匹配)"""

    # 单词到数字的映射表 (处理 "first" vs "1", "two" vs "2" 等常见改写)
    WORD_TO_DIGIT = {
        'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
        'first': '1', 'second': '2', 'third': '3', 'fourth': '4', 'fifth': '5',
        'sixth': '6', 'seventh': '7', 'eighth': '8', 'ninth': '9', 'tenth': '10'
    }

    @staticmethod
    def standardize_text(text: str) -> str:
        """标准化文本 (增强 LaTeX/Markdown 处理 + 数字归一化 + 数学符号清洗)"""
        if not text: return ""
        text = text.lower()
        
        # 0. 预处理 Markdown 标记 (防止 **word** 变成 word 后连在一起，或者变成空格)
        text = text.replace('**', ' ').replace('__', ' ').replace('``', ' ')

        # 1. LaTeX 深度清洗
        for _ in range(2):
            text = re.sub(r'\\(?:text|textbf|textit|mathrm|mathbf|bm)\{([^}]+)\}', r'\1', text)
        
        # 替换常见 LaTeX 符号为标准字符（随后会被 chars_to_space 清洗，或者保留用于数字上下文）
        text = text.replace(r'\times', '×').replace(r'\cdot', '·').replace(r'\div', '÷')
        text = text.replace(r'\pm', '±').replace(r'\approx', '≈').replace(r'\neq', '≠')
        text = text.replace(r'\leq', '≤').replace(r'\geq', '≥')
        text = text.replace(r'\$', '$').replace(r'\%', '%').replace(r'\&', '&').replace(r'\_', '_')
        text = text.replace('$$', ' ').replace('$', ' ') 
        text = text.replace(r'\[', ' ').replace(r'\]', ' ')
        text = text.replace(r'\(', ' ').replace(r'\)', ' ')

        # 2. 统一标点
        text = text.replace('–', '-').replace('—', '-').replace('−', '-') # 统一减号
        text = text.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"')

        # 3. 序数词/数字词归一化
        def replace_match(match):
            word = match.group(0)
            return HallucinationVerifier.WORD_TO_DIGIT.get(word, word)
        
        pattern = r'\b(' + '|'.join(HallucinationVerifier.WORD_TO_DIGIT.keys()) + r')\b'
        text = re.sub(pattern, replace_match, text)

        # 4. 激进清洗 (扩展了清洗列表，包含数学比较符和计算符)
        chars_to_space = [
            '*', '_', '`', '#', '~', '|', '(', ')', '[', ']', '{', '}', '"', "'",
            '\\', ';', '!', '?', '£', '€', '¥', '₹', '\n', '\t', ',', ':', 
            '“', '”', '‘', '’',
            # [新增] 数学符号，防止 "Change =" vs "Change" 或 "x" vs "×" 导致匹配失败
            '=', '×', '÷', '≈', '≠', '≤', '≥', '<', '>', '+', '·' 
            # 注意：保留 '-' 因为它可能表示负数，但在后续 token 匹配中会灵活处理
        ]
        
        for char in chars_to_space:
            text = text.replace(char, ' ')
        
        # 5. 优化句号处理
        text = re.sub(r'\.(?!\d)', ' ', text)
        
        return " ".join(text.split())

    @staticmethod
    def _check_spliced_match(cleaned_quote: str, cleaned_resp: str) -> bool:
        """检查拼接匹配"""
        if "..." in cleaned_quote:
            parts = cleaned_quote.split("...")
            current_idx = 0
            all_parts_found = True
            for part in parts:
                part = part.strip()
                if len(part) < 3: continue 
                idx = cleaned_resp.find(part, current_idx)
                if idx == -1:
                    all_parts_found = False
                    break
                current_idx = idx + len(part)
            if all_parts_found: return True

        if len(cleaned_quote) > 20: 
            head = cleaned_quote[:min(len(cleaned_quote)//3, 20)]
            tail = cleaned_quote[-min(len(cleaned_quote)//3, 20):]
            head_idx = cleaned_resp.find(head)
            if head_idx != -1:
                tail_idx = cleaned_resp.find(tail, head_idx + len(head))
                if tail_idx != -1: return True
        return False

    @staticmethod
    def _check_plus_minus_interval(quote_text: str, pred_interval: List[float]) -> bool:
        """检查 'Value ± Margin' 形式的区间"""
        target_symbol = None
        if '±' in quote_text: target_symbol = '±'
        elif '+/-' in quote_text: target_symbol = '+/-'
        
        if not target_symbol:
            return False

        try:
            parts = quote_text.split(target_symbol)
            if len(parts) != 2: return False
            
            center_str = parts[0].strip()
            margin_str = parts[1].strip()
            
            centers = NumberParser.parse_text_number_to_values(center_str)
            margins = NumberParser.parse_text_number_to_values(margin_str)
            
            if not centers or not margins: return False
            
            pred_min = float(pred_interval[0])
            pred_max = float(pred_interval[1])
            
            for c in centers:
                for m in margins:
                    derived_min = c - m
                    derived_max = c + m
                    if (math.isclose(pred_min, derived_min, rel_tol=0.01) and 
                        math.isclose(pred_max, derived_max, rel_tol=0.01)):
                        return True
                        
        except Exception:
            return False
            
        return False
    
    @staticmethod
    def _check_token_subsequence(quote_text: str, resp_text: str, threshold: float = 0.75) -> bool:
        """
        [增强版] 模糊单词级子序列匹配
        允许不连续、允许少量错词、允许单复数差异 (increase vs increases)
        """
        q_tokens = quote_text.split()
        r_tokens = resp_text.split()
        
        if not q_tokens: return True
        
        # 引用太短时，要求极高的匹配度
        if len(q_tokens) < 3:
            # 简单的包含检查
            return quote_text in resp_text
            
        matches = 0
        current_resp_idx = 0
        
        for q_word in q_tokens:
            # 在剩余的 response tokens 中寻找匹配
            # 允许简单的模糊匹配：
            # 1. 完全相等
            # 2. 一个包含另一个 (处理 increase/increases, change/changed)
            # 3. 相似度极高 (处理 typo)
            
            found = False
            # 限制搜索窗口，防止跨度过大导致误匹配 (例如向后搜索50个词)
            search_window = 50 
            end_search = min(current_resp_idx + search_window, len(r_tokens))
            
            for i in range(current_resp_idx, end_search):
                r_word = r_tokens[i]
                
                # 匹配逻辑
                is_match = False
                if q_word == r_word:
                    is_match = True
                elif len(q_word) > 3 and len(r_word) > 3:
                    if q_word in r_word or r_word in q_word: # 包含关系 (tense/plural)
                        is_match = True
                    else:
                        # 昂贵但有效的相似度检查
                        if difflib.SequenceMatcher(None, q_word, r_word).ratio() > 0.85:
                            is_match = True
                
                if is_match:
                    matches += 1
                    current_resp_idx = i + 1 # 推进指针
                    found = True
                    break
            
            if not found:
                # 没找到该词，继续找下一个引用词
                continue
        
        coverage = matches / len(q_tokens)
        return coverage >= threshold
    
    @staticmethod
    def _check_entailment_with_llm(context: str, claim: str) -> Tuple[bool, str]:
        """
        [新增] 使用 LLM 进行自然语言推理 (NLI) 检查蕴含关系
        返回: (是否蕴含, 修改建议/理由)
        """
        # 截断 context 防止 token 溢出，只保留相关部分会更好，但这里简单截断
        safe_context = context[:] 
        
        prompt = f"""
You are a strict Fact-Checking Judge. 
Task: Determine if the [Claim] is explicitly supported or entailed by the [Source Text].

[Source Text]:
{safe_context}

[Claim (extracted quote)]:
{claim}

INSTRUCTIONS:
1. **Entailment Check**: Does the content of the Claim logically exist in the Source Text?
   - It DOES NOT need to be a verbatim string match.
   - It DOES need to carry the exact same meaning (summarization or paraphrasing is allowed IF the facts remain 100% accurate).
   - Numeric values must be mathematically equivalent (e.g., "1.5k" entails "1500").
   - If the Claim contains information NOT present in the Source, output "is_entailed": false.

2. **Feedback**: If it is NOT entailed, provide a specific instruction on how to fix the extraction.

OUTPUT JSON FORMAT:
{{
  "is_entailed": boolean,
  "reasoning": "brief explanation",
  "correction_instruction": "Instruction for the model to re-extract correctly (if false)"
}}
"""
        try:
            # 复用 APIClient (使用 deepseek-chat)
            completion = APIClient.generate_completion_with_retry(
                model=JUDGE_MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, # 必须为 0 以保证确定性
                response_format={"type": "json_object"}
            )
            raw_content = completion.choices[0].message.content
            parsed = json.loads(extract_json_content(raw_content))
            
            is_entailed = parsed.get("is_entailed", False)
            reasoning = parsed.get("reasoning", "No reasoning provided")
            instruction = parsed.get("correction_instruction", "Please re-verify the quote against the source text.")
            
            # 如果不蕴含，返回 (False, 建议)
            if not is_entailed:
                return False, f"NLI Check Failed: {reasoning}. Suggestion: {instruction}"
            
            return True, "Entailed via NLI"
            
        except Exception as e:
            logger.warning(f"NLI Check Exception: {e}")
            # 如果 NLI 失败，保守起见认为验证不通过
            return False, f"NLI verification encountered an error: {str(e)}"
        
    @staticmethod
    def _check_number_match_with_llm(quote_text: str, target_val: float) -> Tuple[bool, str]:
        """
        [新增] 当正则无法从文本中解析出目标数值时，使用 LLM 进行数学/语义判断
        返回: (是否匹配, 理由/修改建议)
        """
        prompt = f"""
You are a strict Math Judge.
Task: Determine if the [Target Number] is semantically present in the [Source Text].

[Source Text]: "{quote_text}"
[Target Number]: {target_val}

INSTRUCTIONS:
1. **Equivalence Check**: 
   - Does the text contain a number that equals the Target Number?
   - Handle units: "1.5k" == 1500, "2m" == 2000000, "50%" == 50 (if treating percentage as value) or 0.5.
   - Handle rounding: "approx 3.33" ~= 3.33333 (Accept minor rounding differences).
   - Handle text numbers: "twenty-five" == 25.
   - If the number is NOT found, or completely different, output false.

2. **Feedback**: If false, provide the EXACT number found in the text (if any) or a correction instruction for the number extraction.

OUTPUT JSON FORMAT:
{{
  "is_match": boolean,
  "reasoning": "Brief explanation",
  "correction_instruction": "Instruction to fix the extraction (e.g., 'Text says 1.2, not 1.5')"
}}
"""
        try:
            completion = APIClient.generate_completion_with_retry(
                model=JUDGE_MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            raw_content = completion.choices[0].message.content
            parsed = json.loads(extract_json_content(raw_content))
            
            is_match = parsed.get("is_match", False)
            instruction = parsed.get("correction_instruction", "Number not found in quote.")
            
            if not is_match:
                return False, f"LLM Math Check Failed: {instruction}"
            return True, "Matched via LLM"
            
        except Exception as e:
            logger.warning(f"Number Check Exception: {e}")
            return False, f"Number verification error: {str(e)}"
        
    @staticmethod
    def _check_interval_bound_with_llm(quote_text: str, target_val: float, bound_type: str) -> Tuple[bool, str]:
        """
        [新增] 专门用于检查区间边界 (Lower/Upper)
        bound_type: 'lower' or 'upper'
        """
        if bound_type not in ['lower', 'upper']:
            return False, "Invalid bound type"

        type_desc = "LOWER bound (minimum value)" if bound_type == "lower" else "UPPER bound (maximum value)"
        
        prompt = f"""
You are a strict Math Judge specializing in Interval Analysis.
Task: Determine if the number {target_val} is explicitly the {type_desc} of the interval/range described in the [Source Text].

[Source Text]: "{quote_text}"
[Target {bound_type.capitalize()} Bound]: {target_val}

INSTRUCTIONS:
1. **Strict Boundary Check**: 
   - The text must support that {target_val} is the {bound_type} end of the range.
   - Handle units: "10k" == 10000, "50%" == 0.5.
   - Handle phrases: 
     - "from 10 to 20" -> 10 is Lower, 20 is Upper.
     - "more than 100" -> 100 is Lower.
     - "up to 500" -> 500 is Upper.

2. **Feedback**: If the number is present but is NOT the {bound_type} bound (e.g., you are checking Lower but it is the Upper bound), return false. If false, provide the EXACT number found in the text (if any) or a correction instruction for the number extraction.

OUTPUT JSON FORMAT:
{{
  "is_match": boolean,
  "reasoning": "Explain why it matches or fails as the {bound_type} bound",
  "correction_instruction": "Instruction to fix the extraction"
}}
"""
        try:
            completion = APIClient.generate_completion_with_retry(
                model=JUDGE_MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            raw_content = completion.choices[0].message.content
            parsed = json.loads(extract_json_content(raw_content))
            
            is_match = parsed.get("is_match", False)
            instruction = parsed.get("correction_instruction", f"Value not confirmed as {bound_type} bound.")
            
            if not is_match:
                return False, f"LLM Interval {bound_type.capitalize()} Check Failed: {instruction}"
            return True, f"Matched {bound_type} bound via LLM"
            
        except Exception as e:
            logger.warning(f"Interval Bound Check Exception: {e}")
            return False, f"Interval verification error: {str(e)}"

    @staticmethod
    def verify(response_text: str, extracted_payload: Dict, task_type: str = "regression", context_prefix: str = "") -> Tuple[bool, Optional[str]]:
        if not response_text or not extracted_payload:
            return True, None
        
        err_prefix = f"[{context_prefix.strip()}] " if context_prefix and context_prefix.strip() else ""
        clean_resp = HallucinationVerifier.standardize_text(response_text)
        
        def check_quote_existence(quote_text, field_name):
            if not quote_text: return True, None
            clean_q = HallucinationVerifier.standardize_text(quote_text)
            
            if clean_q in clean_resp: return True, None
            if HallucinationVerifier._check_spliced_match(clean_q, clean_resp): return True, None

            matcher = difflib.SequenceMatcher(None, clean_q, clean_resp)
            match = matcher.find_longest_match(0, len(clean_q), 0, len(clean_resp))
            if match.size > len(clean_q) * 0.80 or matcher.ratio() > 0.80: return True, None
            
            # 数字锚点宽松匹配
            quote_nums = re.findall(r'\d+', clean_q)
            if quote_nums and all(num in clean_resp for num in quote_nums):
                if matcher.ratio() > 0.6: return True, None
                if match.size > 0:
                    matched_str = clean_q[match.a : match.a + match.size]
                    if any(num in matched_str for num in quote_nums) and match.size > len(clean_q) * 0.5:
                        return True, None
                    
            if HallucinationVerifier._check_token_subsequence(clean_q, clean_resp, threshold=0.80):
                    return True, None
            
            logger.info(f"{context_prefix}Substring match failed for '{field_name}', invoking DeepSeek NLI check...")
            is_entailed, feedback_msg = HallucinationVerifier._check_entailment_with_llm(response_text, quote_text)

            if is_entailed:
                return True, None
            
            # NLI 也不通过，返回反馈意见给 Judge Model 进行重试
            # return False, f"{err_prefix}{field_name} is NOT found in source text verbatim, and NLI check failed. {feedback_msg}"
                
            return False, f"{err_prefix}{field_name} '{quote_text}' was not found and NLI check failed. The original response text:\n'{response_text}'. \nAdvices:\n{feedback_msg}"

        if task_type == "regression":
            # 1. 验证点值
            pred_val = extracted_payload.get('predicted_value')
            quote_val = extracted_payload.get('proof_quote_value')
            
            if pred_val is not None:
                if not quote_val:
                    return False, f"{err_prefix}The predicted_value '{pred_val}' was extracted, but the proof_quote_value was not provided."
                ok, msg = check_quote_existence(quote_val, "proof_quote_value")
                if not ok: return False, msg
                
                valid_nums_val = NumberParser.parse_text_number_to_values(quote_val)
                if not NumberParser.is_value_match(pred_val, valid_nums_val):
                    logger.info(f"{context_prefix}Regex failed matching {pred_val} in '{quote_val}', invoking LLM Math Check...")
                    is_num_match, num_feedback = HallucinationVerifier._check_number_match_with_llm(quote_val, float(pred_val))
                    
                    if not is_num_match:
                        return False, f"{err_prefix}predicted_value {pred_val} NOT confirmed in quote '{quote_val}'. \nAdvices:\n{num_feedback}"
                    # return False, f"{err_prefix}predicted_value {pred_val} was not found in the proof_quote_value '{quote_val}'"

            # 2. 验证区间
            pred_interval = extracted_payload.get('predicted_interval')
            quote_int = extracted_payload.get('proof_quote_interval')
            
            if pred_interval and isinstance(pred_interval, list) and len(pred_interval) == 2:
                if not quote_int:
                     return False, f"{err_prefix}predicted_interval '{pred_interval}' exists but proof_quote_interval is missing."
                ok, msg = check_quote_existence(quote_int, "proof_quote_interval")
                if not ok: return False, msg
                
                # A. 常规端点检查
                valid_nums_int = NumberParser.parse_text_number_to_values(quote_int)
                lower_match = NumberParser.is_value_match(pred_interval[0], valid_nums_int)
                upper_match = NumberParser.is_value_match(pred_interval[1], valid_nums_int)

                # 检查 Lower Bound
                lower_val = pred_interval[0]
                if lower_val is not None and not NumberParser.is_value_match(lower_val, valid_nums_int):
                    # [修改] 使用新的专用 Interval Bound Check (Lower)
                    is_match, feedback = HallucinationVerifier._check_interval_bound_with_llm(quote_int, float(lower_val), "lower")
                    if not is_match:
                         return False, f"{err_prefix}Interval Lower Bound {lower_val} not found in proof_quote_interval '{quote_int}'. \nAdvices:\n{feedback}"

                # 检查 Upper Bound
                upper_val = pred_interval[1]
                if upper_val is not None and not NumberParser.is_value_match(upper_val, valid_nums_int):
                    # [修改] 使用新的专用 Interval Bound Check (Upper)
                    is_match, feedback = HallucinationVerifier._check_interval_bound_with_llm(quote_int, float(upper_val), "upper")
                    if not is_match:
                         # 最后尝试一次 ± 推导作为兜底
                         if not HallucinationVerifier._check_plus_minus_interval(quote_int, pred_interval):
                             return False, f"{err_prefix}Interval Upper Bound {upper_val} not found in proof_quote_interval '{quote_int}'. \nAdvices:\n{feedback}"
                
                # if not (lower_match and upper_match):
                #     # B. [新增] 尝试 "Center ± Margin" 推导检查
                #     if HallucinationVerifier._check_plus_minus_interval(quote_int, pred_interval):
                #         return True, None
                        
                #     return False, f"{err_prefix}The endpoint of predicted_interval {pred_interval} was not found in proof_quote_interval '{quote_int}'"

        elif task_type == "classification":
            pred_cat = extracted_payload.get('predicted_category')
            proof_quote = extracted_payload.get('proof_quote', "")
            
            if pred_cat is not None and not proof_quote:
                 return False, f"{err_prefix}The predicted_category '{pred_cat}' was extracted, but the proof_quote was not provided."
            if proof_quote:
                ok, msg = check_quote_existence(proof_quote, "proof_quote")
                if not ok: return False, msg

        return True, None

# ================= 统计缓存 =================
class StatsCache:
    """数据集统计缓存管理器 (增强版：支持分位数统计)"""
    
    @staticmethod
    def compute_and_cache_stats(base_dir: str) -> Dict:
        """计算并缓存数据集统计信息，增加 Q0.05 和 Q0.95 以应对离群点"""
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
        for fname in ["history.csv", "test.csv"]:
            fpath = os.path.join(base_dir, fname)
            if os.path.exists(fpath):
                try:
                    # low_memory=False 保证类型推断更准确
                    df_temp = pd.read_csv(fpath, on_bad_lines='skip', encoding_errors='ignore', low_memory=False)
                    dfs.append(df_temp)
                    logger.info(f"加载CSV: {fname}, 形状: {df_temp.shape}")
                except Exception as e:
                    logger.error(f"读取CSV失败 {fname}: {e}")
        
        if not dfs:
            logger.warning(f"未找到有效的CSV文件: {base_dir}")
            return {}
        
        full_df = pd.concat(dfs, ignore_index=True)
        stats = {}
        
        for col in full_df.columns:
            series = full_df[col]
            clean_series = series.dropna()
            
            if clean_series.empty:
                continue
            
            col_stat = {}
            
            # 数值类型处理
            if pd.api.types.is_numeric_dtype(series):
                desc = clean_series.describe()
                col_stat["type"] = "numeric"
                col_stat["std"] = float(desc.get("std", 0.0))
                col_stat["mean"] = float(desc.get("mean", 0.0))
                col_stat["min"] = float(desc.get("min", 0.0))
                col_stat["max"] = float(desc.get("max", 0.0))
                
                # --- [核心修改] 添加分位数统计 ---
                # 使用 5% 和 95% 分位数来定义“稳健范围 (Robust Range)”
                # 这能有效过滤掉表格数据中常见的极端错误值（如 99999999 或 -1 等噪音）
                try:
                    quantiles = clean_series.quantile([0.05, 0.95]).to_dict()
                    col_stat["q05"] = float(quantiles.get(0.05, col_stat["min"]))
                    col_stat["q95"] = float(quantiles.get(0.95, col_stat["max"]))
                except Exception as e:
                    logger.warning(f"计算列 {col} 分位数失败: {e}")
                    col_stat["q05"] = col_stat["min"]
                    col_stat["q95"] = col_stat["max"]
                
                # 离散数值检查
                unique_vals = clean_series.unique()
                if len(unique_vals) < 50:
                    col_stat["discrete_values"] = sorted([float(x) for x in unique_vals])
            
            # 类别/文本类型处理 (保持你之前的优化逻辑)
            else:
                unique_vals = clean_series.unique()
                n_unique = len(unique_vals)
                n_total = len(clean_series)
                
                if n_unique == n_total or (n_total > 0 and n_unique / n_total > 0.8):
                    col_stat["type"] = "string"
                    col_stat["cardinality"] = n_unique
                else:
                    col_stat["type"] = "categorical"
                    col_stat["categories"] = sorted([str(x) for x in unique_vals])
                    col_stat["cardinality"] = n_unique
            
            stats[col] = col_stat
        
        if not DISABLE_STATS_CACHE:
            safe_json_dump(stats, cache_path)
            logger.info(f"统计信息已更新并缓存至: {cache_path}")
        
        return stats
    
from functools import lru_cache
class DatasetContextLoader:
    """数据集上下文加载器"""
    
    @staticmethod
    @lru_cache(maxsize=32)
    def load_dataset_context(base_dir: str) -> Dict:
        """加载数据集元信息和统计数据"""
        context = {
            "dataset_std": None,
            "valid_categories": None,
            "target_col_hint": None,
            "task_type_hint": "regression",
            "stats": {},
            "prompt_meta_str": ""
        }
        
        # 加载info.json
        info = safe_json_load(os.path.join(base_dir, "info.json")) or {}
        
        # 提取目标列信息
        target_col, target_desc = DatasetContextLoader._extract_target_info(info)
        task_type = info.get("task_type", "regression")
        
        context["target_col_hint"] = target_col
        context["task_type_hint"] = task_type
        
        # 加载统计数据
        real_stats = StatsCache.compute_and_cache_stats(base_dir)
        context["stats"] = real_stats
        
        # 预填充关键统计指标
        if target_col and target_col in real_stats:
            t_stat = real_stats[target_col]
            if t_stat["type"] == "numeric":
                context["dataset_std"] = t_stat.get("std")
            elif t_stat["type"] == "categorical":
                context["valid_categories"] = (
                    t_stat.get("discrete_values") or 
                    t_stat.get("categories")
                )
        
        # 构建Prompt元信息
        context["prompt_meta_str"] = DatasetContextLoader._build_prompt_meta(
            target_col, target_desc, task_type, real_stats, info
        )
        
        return context
    
    @staticmethod
    def _extract_target_info(info: Dict) -> Tuple[Optional[str], str]:
        """提取目标列名称和描述"""
        raw_target = info.get("target")
        target_col = None
        target_desc = ""
        
        # 处理字典格式
        if isinstance(raw_target, dict) and raw_target:
            target_col = next(iter(raw_target))
            target_desc = str(raw_target[target_col])
        # 处理字符串格式
        elif raw_target:
            target_col = str(raw_target).strip()
        
        # 兜底查找描述
        if target_col and not target_desc:
            num_intro = info.get("num_feature_intro", {})
            cat_intro = info.get("cat_feature_intro", {})
            target_desc = num_intro.get(target_col) or cat_intro.get(target_col) or ""
        
        return target_col, target_desc
    
    @staticmethod
    def _build_prompt_meta(target_col: str, target_desc: str, task_type: str, 
                          stats: Dict, info: Dict) -> str:
        """构建Prompt元信息字符串"""
        lines = []
        
        # 目标列信息
        target_line = f"- Target Column: '{target_col}' (Task Type: {task_type})"
        if target_desc:
            clean_desc = str(target_desc).replace(f"{target_col}:", "").strip()
            target_line += f" | Goal: {clean_desc}"
        lines.append(target_line)
        
        # 特征信息
        if stats:
            lines.append("\n- Features Overview:")
            num_intro = info.get("num_feature_intro", {})
            cat_intro = info.get("cat_feature_intro", {})
            
            for col, stat in stats.items():
                # if col == target_col:
                #     continue
                
                # 获取特征描述
                raw_desc = num_intro.get(col) or cat_intro.get(col) or ""
                clean_desc = str(raw_desc).replace(f"- {col}:", "").replace(f"{col}:", "").strip()
                desc_part = f"Desc: {clean_desc} | " if clean_desc else ""
                
                if stat["type"] == "numeric":
                    rng_str = f"[{stat['min']:.2f}, {stat['max']:.2f}]"
                    lines.append(
                        f"  * {col} (Numeric): {desc_part}"
                    )
                    if col == target_col:
                        lines.append(f"Range {rng_str}, Mean {stat['mean']:.2f}")
                else:
                    cats = stat.get("categories", [])
                    if len(cats) > 15:
                        cat_str = f"{str(cats[:15])[:-1]}, ... total {len(cats)}]"
                    else:
                        cat_str = str(cats)
                    # lines.append(f"  * {col} (Categorical): {desc_part}Values: {cat_str}")
                    lines.append(f"  * {col} (Categorical): {desc_part}")
                    if col == target_col:
                        lines.append(f"Values: {cat_str}")
                
        
        return "\n".join(lines)


def get_scale_bucket(row_count: Optional[int]) -> str:
    if row_count is None:
        return "unknown"
    short_max, medium_max = SCALE_THRESHOLDS
    if row_count <= short_max:
        return "short"
    if row_count <= medium_max:
        return "medium"
    return "long"


def get_column_bucket(column_count: Optional[int]) -> str:
    if column_count is None:
        return "unknown"
    if column_count <= 10:
        return "narrow"
    if column_count <= 30:
        return "mid"
    return "wide"


@lru_cache(maxsize=512)
def load_dataset_profile(dataset_path: str, benchmark_type: str) -> Dict[str, Any]:
    dataset_dir = Path(dataset_path)
    info_path = dataset_dir / "info.json"
    info = safe_json_load(str(info_path)) or {}

    history_path = dataset_dir / "history.csv"
    row_count = None
    column_count = None
    if history_path.exists():
        try:
            header_df = pd.read_csv(history_path, nrows=0, low_memory=False)
            column_count = int(len(header_df.columns))
        except Exception:
            column_count = None
        try:
            if "overall_size" in info and str(info.get("overall_size", "")).strip().isdigit():
                row_count = int(info["overall_size"])
            else:
                row_count = int(sum(1 for _ in open(history_path, "r", encoding="utf-8", errors="ignore")) - 1)
        except Exception:
            row_count = None

    task_sub_type = str(info.get("task_type") or "").strip().lower() or "unknown"
    rel_parts = dataset_dir.relative_to(PATH_CONFIG.dataset_root).parts if dataset_dir.is_relative_to(PATH_CONFIG.dataset_root) else dataset_dir.parts
    domain = rel_parts[0] if rel_parts else "unknown"

    return {
        "dataset_path": str(dataset_dir),
        "dataset_name": dataset_dir.name,
        "benchmark": benchmark_type,
        "domain": domain,
        "task_sub_type": task_sub_type,
        "row_count": row_count,
        "column_count": column_count,
        "scale_bucket": get_scale_bucket(row_count),
        "column_bucket": get_column_bucket(column_count),
        "overall_size": info.get("overall_size"),
        "train_size": info.get("train_size"),
        "test_size": info.get("test_size"),
    }


def flatten_eval_record(eval_json: Dict[str, Any], inference_path: str, model_name: str, mode: str, dataset_profile: Dict[str, Any]) -> Dict[str, Any]:
    breakdown = eval_json.get("breakdown", {}) or {}
    record = {
        "inference_path": inference_path,
        "eval_path": get_eval_path(inference_path),
        "model_name": model_name,
        "mode": mode,
        "benchmark": eval_json.get("benchmark", PATH_CONFIG.target_benchmark),
        "task_type": eval_json.get("task_type", ""),
        "final_score": float(eval_json.get("final_score", 0.0) or 0.0),
        "accuracy_score": None,
        "logic_score": None,
        "decision_score": None,
        "trend_score": None,
        **dataset_profile,
    }

    if "accuracy" in breakdown:
        record["accuracy_score"] = float((breakdown.get("accuracy") or {}).get("score", 0.0) or 0.0)
        record["logic_score"] = float((breakdown.get("logic") or {}).get("score", 0.0) or 0.0)
    elif "avg_prediction" in breakdown:
        record["accuracy_score"] = float((breakdown.get("avg_prediction") or {}).get("score", 0.0) or 0.0)
        record["logic_score"] = float((breakdown.get("logic") or {}).get("score", 0.0) or 0.0)
        record["decision_score"] = float((breakdown.get("decision") or {}).get("score", 0.0) or 0.0)
    elif "pred_002_accuracy" in breakdown:
        record["accuracy_score"] = float((breakdown.get("pred_002_accuracy") or {}).get("score", 0.0) or 0.0)
        record["logic_score"] = float((breakdown.get("logic") or {}).get("score", 0.0) or 0.0)
        record["trend_score"] = float((breakdown.get("trend_accuracy") or {}).get("score", 0.0) or 0.0)

    return record


def summarize_records_df(records_df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if records_df.empty:
        return pd.DataFrame()

    summary_rows = []
    for keys, group in records_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        row["count"] = int(len(group))
        row["avg_final"] = round(float(group["final_score"].mean()), 4)
        row["avg_accuracy"] = round(float(group["accuracy_score"].fillna(0.0).mean()), 4)
        row["avg_logic"] = round(float(group["logic_score"].fillna(0.0).mean()), 4)
        if "decision_score" in group.columns and group["decision_score"].notna().any():
            row["avg_decision"] = round(float(group["decision_score"].dropna().mean()), 4)
        if "trend_score" in group.columns and group["trend_score"].notna().any():
            row["avg_trend"] = round(float(group["trend_score"].dropna().mean()), 4)
        if "row_count" in group.columns and group["row_count"].notna().any():
            row["avg_rows"] = round(float(group["row_count"].dropna().mean()), 2)
        if "column_count" in group.columns and group["column_count"].notna().any():
            row["avg_columns"] = round(float(group["column_count"].dropna().mean()), 2)
        summary_rows.append(row)
    return pd.DataFrame(summary_rows).sort_values(group_cols).reset_index(drop=True)


def recompute_regression_score_from_details(
    details: Dict[str, Any],
    width_factor: float,
    alpha: float,
    point_weight: float = 0.6,
    interval_weight: float = 0.4,
) -> Optional[float]:
    if not isinstance(details, dict):
        return None
    if "nmae_score" not in details or "overlap_score" not in details:
        return None
    pred_range = details.get("pred_range") or []
    gt_range = details.get("gt_range") or []
    if len(pred_range) != 2 or len(gt_range) != 2:
        return None
    try:
        pred_width = float(pred_range[1]) - float(pred_range[0])
        gt_width = max(float(gt_range[1]) - float(gt_range[0]), 1e-9)
        width_ratio = pred_width / gt_width
        penalty = 1.0
        if width_ratio > width_factor:
            penalty = max(0.0, math.exp(-alpha * (width_ratio - width_factor)))
        base_score = point_weight * float(details["nmae_score"]) + interval_weight * float(details["overlap_score"])
        return round(max(0.0, min(1.0, base_score * penalty)), 4)
    except Exception:
        return None


def _format_weight_token(value: float) -> str:
    return str(int(round(value * 100)))


def _dedupe_profiles(profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for profile in profiles:
        name = str(profile.get("name") or "")
        if not name or name in seen:
            continue
        seen.add(name)
        deduped.append(profile)
    return deduped


def get_metric_sensitivity_profiles(benchmark_type: str, full_grid: bool = False) -> List[Dict[str, Any]]:
    profiles = [{"name": "default", "kind": benchmark_type}]

    if not full_grid:
        base_name = benchmark_type.lower()
        profiles.extend([
            {"name": f"{base_name}_point_heavy", "kind": benchmark_type, "point_weight": 0.8, "interval_weight": 0.2, "width_factor": 2.0, "width_alpha": 1.0},
            {"name": f"{base_name}_interval_heavy", "kind": benchmark_type, "point_weight": 0.4, "interval_weight": 0.6, "width_factor": 2.0, "width_alpha": 1.0},
            {"name": f"{base_name}_interval_lenient", "kind": benchmark_type, "point_weight": 0.6, "interval_weight": 0.4, "width_factor": 3.0, "width_alpha": 0.5},
            {"name": f"{base_name}_interval_strict", "kind": benchmark_type, "point_weight": 0.6, "interval_weight": 0.4, "width_factor": 1.5, "width_alpha": 2.0},
        ])
    else:
        base_name = benchmark_type.lower()
        for point_weight in METRIC_WEIGHT_GRID:
            interval_weight = round(1.0 - point_weight, 4)
            if interval_weight <= 0:
                continue
            for width_factor in METRIC_WIDTH_FACTORS:
                for width_alpha in METRIC_WIDTH_ALPHAS:
                    profiles.append({
                        "name": f"{base_name}_pw{_format_weight_token(point_weight)}_iw{_format_weight_token(interval_weight)}_wf{str(width_factor).replace('.', 'p')}_a{str(width_alpha).replace('.', 'p')}",
                        "kind": benchmark_type,
                        "point_weight": point_weight,
                        "interval_weight": interval_weight,
                        "width_factor": width_factor,
                        "width_alpha": width_alpha,
                    })
    return _dedupe_profiles(profiles)


def load_per_file_analysis_records(inference_files: List[Tuple[str, str, str]]) -> List[Dict[str, Any]]:
    records = []
    for inference_path, model_name, mode in inference_files:
        eval_path = get_eval_path(inference_path)
        eval_json = safe_json_load(eval_path)
        if not eval_json:
            continue
        dataset_path = PathManager.get_dataset_path_from_inference(
            inference_path,
            PATH_CONFIG.inference_root,
            PATH_CONFIG.dataset_root
        )
        dataset_profile = load_dataset_profile(dataset_path, PATH_CONFIG.target_benchmark)
        record = flatten_eval_record(eval_json, inference_path, model_name, mode, dataset_profile)
        records.append(record)
    return records


def export_scale_breakdown(records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    report_dir = os.path.join(PATH_CONFIG.inference_root, "_analysis")
    os.makedirs(report_dir, exist_ok=True)
    df = pd.DataFrame(records)
    full_path = os.path.join(report_dir, f"scale_breakdown_records_{PATH_CONFIG.target_benchmark}.csv")
    df.to_csv(full_path, index=False)

    summary_df = summarize_records_df(df, ["mode", "model_name", "scale_bucket"])
    summary_path = os.path.join(report_dir, f"scale_breakdown_summary_{PATH_CONFIG.target_benchmark}.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"已导出规模分层结果: {summary_path}")


def export_shape_breakdown(records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    report_dir = os.path.join(PATH_CONFIG.inference_root, "_analysis")
    os.makedirs(report_dir, exist_ok=True)
    df = pd.DataFrame(records)
    if "column_bucket" not in df.columns:
        df["column_bucket"] = df["column_count"].apply(get_column_bucket)

    df.to_csv(os.path.join(report_dir, f"shape_breakdown_records_{PATH_CONFIG.target_benchmark}.csv"), index=False)

    shape_summary_df = summarize_records_df(df, ["mode", "model_name", "scale_bucket", "column_bucket"])
    shape_summary_df.to_csv(os.path.join(report_dir, f"shape_breakdown_summary_{PATH_CONFIG.target_benchmark}.csv"), index=False)

    domain_shape_df = summarize_records_df(df, ["mode", "model_name", "domain", "scale_bucket", "column_bucket"])
    domain_shape_df.to_csv(os.path.join(report_dir, f"shape_breakdown_by_domain_{PATH_CONFIG.target_benchmark}.csv"), index=False)
    logger.info(f"已导出表形状分层结果: {report_dir}")


def _spearman_corr_from_ranks(rank_a: Dict[str, float], rank_b: Dict[str, float]) -> Optional[float]:
    common = sorted(set(rank_a) & set(rank_b))
    if len(common) < 2:
        return None
    a = pd.Series([rank_a[m] for m in common], dtype=float)
    b = pd.Series([rank_b[m] for m in common], dtype=float)
    return float(a.corr(b, method="spearman"))


def export_metric_rank_stability(raw_df: pd.DataFrame, report_dir: str, benchmark_type: str, suffix: str = "") -> None:
    if raw_df.empty:
        return
    summary_df = summarize_records_df(raw_df, ["profile", "mode", "model_name"])
    if summary_df.empty:
        return

    rows = []
    for mode, mode_group in summary_df.groupby("mode", dropna=False):
        default_group = mode_group.loc[mode_group["profile"] == "default"].copy()
        if default_group.empty:
            continue
        default_group["rank"] = default_group["avg_final"].rank(method="dense", ascending=False)
        default_rank = dict(zip(default_group["model_name"], default_group["rank"]))
        default_order = default_group.sort_values(["rank", "model_name"])["model_name"].tolist()
        default_top1 = default_order[0] if default_order else None

        for profile, profile_group in mode_group.groupby("profile", dropna=False):
            profile_group = profile_group.copy()
            profile_group["rank"] = profile_group["avg_final"].rank(method="dense", ascending=False)
            profile_rank = dict(zip(profile_group["model_name"], profile_group["rank"]))
            ordered = profile_group.sort_values(["rank", "model_name"])["model_name"].tolist()
            top1 = ordered[0] if ordered else None
            common_models = sorted(set(default_rank) & set(profile_rank))
            pairwise_total = 0
            pairwise_same = 0
            for i, left in enumerate(common_models):
                for right in common_models[i + 1:]:
                    pairwise_total += 1
                    default_rel = np.sign(default_rank[left] - default_rank[right])
                    profile_rel = np.sign(profile_rank[left] - profile_rank[right])
                    if default_rel == profile_rel:
                        pairwise_same += 1
            rows.append({
                "benchmark": benchmark_type,
                "mode": mode,
                "profile": profile,
                "model_count": len(common_models),
                "default_top1": default_top1,
                "profile_top1": top1,
                "top1_same_as_default": int(default_top1 == top1) if default_top1 and top1 else None,
                "spearman_rank_corr_vs_default": _spearman_corr_from_ranks(default_rank, profile_rank),
                "pairwise_order_agreement_vs_default": (pairwise_same / pairwise_total) if pairwise_total else None,
            })

    rank_df = pd.DataFrame(rows)
    if rank_df.empty:
        return
    rank_df.to_csv(os.path.join(report_dir, f"metric_rank_stability_{benchmark_type}{suffix}.csv"), index=False)
    overall = rank_df.groupby(["benchmark", "profile"], dropna=False).agg(
        mode_count=("mode", "count"),
        avg_spearman_rank_corr_vs_default=("spearman_rank_corr_vs_default", "mean"),
        avg_pairwise_order_agreement_vs_default=("pairwise_order_agreement_vs_default", "mean"),
        top1_same_rate=("top1_same_as_default", "mean"),
    ).reset_index()
    overall.to_csv(os.path.join(report_dir, f"metric_rank_stability_overall_{benchmark_type}{suffix}.csv"), index=False)


def export_metric_sensitivity(inference_files: List[Tuple[str, str, str]], full_grid: bool = False) -> None:
    if not inference_files:
        return
    rows = []
    profiles = get_metric_sensitivity_profiles(PATH_CONFIG.target_benchmark, full_grid=full_grid)
    for inference_path, model_name, mode in inference_files:
        eval_json = safe_json_load(get_eval_path(inference_path))
        if not eval_json:
            continue
        breakdown = eval_json.get("breakdown", {}) or {}
        dataset_path = PathManager.get_dataset_path_from_inference(
            inference_path,
            PATH_CONFIG.inference_root,
            PATH_CONFIG.dataset_root
        )
        dataset_profile = load_dataset_profile(dataset_path, PATH_CONFIG.target_benchmark)
        for profile in profiles:
            final_score = None
            acc = None
            logic = None
            decision = None
            avg_pred = None
            trend = None
            pred_002 = None
            if profile["name"] == "default":
                final_score = float(eval_json.get("final_score", 0.0) or 0.0)
                if PATH_CONFIG.target_benchmark == "B1":
                    acc = float((breakdown.get("accuracy") or {}).get("score", 0.0) or 0.0)
                    logic = float((breakdown.get("logic") or {}).get("score", 0.0) or 0.0)
                elif PATH_CONFIG.target_benchmark == "B2":
                    decision = float((breakdown.get("decision") or {}).get("score", 0.0) or 0.0)
                    logic = float((breakdown.get("logic") or {}).get("score", 0.0) or 0.0)
                    avg_pred = float((breakdown.get("avg_prediction") or {}).get("score", 0.0) or 0.0)
                elif PATH_CONFIG.target_benchmark == "B3":
                    trend = float((breakdown.get("trend_accuracy") or {}).get("score", 0.0) or 0.0)
                    logic = float((breakdown.get("logic") or {}).get("score", 0.0) or 0.0)
                    pred_002 = float((breakdown.get("pred_002_accuracy") or {}).get("score", 0.0) or 0.0)
            elif PATH_CONFIG.target_benchmark == "B1":
                acc = float((breakdown.get("accuracy") or {}).get("score", 0.0) or 0.0)
                logic = float((breakdown.get("logic") or {}).get("score", 0.0) or 0.0)
                if "width_factor" in profile:
                    details = (breakdown.get("accuracy") or {}).get("details", {}) or {}
                    adj_acc = recompute_regression_score_from_details(
                        details,
                        profile["width_factor"],
                        profile["width_alpha"],
                        point_weight=profile.get("point_weight", 0.6),
                        interval_weight=profile.get("interval_weight", 0.4),
                    )
                    if adj_acc is not None:
                        acc = adj_acc
                final_score = SCORING_CONFIG.accuracy_weight * acc + SCORING_CONFIG.logic_weight * logic
            elif PATH_CONFIG.target_benchmark == "B2":
                decision = float((breakdown.get("decision") or {}).get("score", 0.0) or 0.0)
                logic = float((breakdown.get("logic") or {}).get("score", 0.0) or 0.0)
                avg_pred = float((breakdown.get("avg_prediction") or {}).get("score", 0.0) or 0.0)
                if "width_factor" in profile:
                    scenario_map = (breakdown.get("scenarios_details") or {})
                    adjusted_scores = []
                    changed = False
                    for scenario in scenario_map.values():
                        if not isinstance(scenario, dict):
                            continue
                        score = scenario.get("score")
                        details = scenario.get("details")
                        adj = recompute_regression_score_from_details(
                            details,
                            profile["width_factor"],
                            profile["width_alpha"],
                            point_weight=profile.get("point_weight", 0.6),
                            interval_weight=profile.get("interval_weight", 0.4),
                        )
                        if adj is None:
                            if score is not None:
                                adjusted_scores.append(float(score))
                        else:
                            adjusted_scores.append(float(adj))
                            changed = True
                    if changed and adjusted_scores:
                        avg_pred = float(sum(adjusted_scores) / len(adjusted_scores))
                final_score = (
                    SCORING_CONFIG.b2_decision_weight * decision
                    + SCORING_CONFIG.b2_avg_pred_weight * avg_pred
                    + SCORING_CONFIG.b2_logic_weight * logic
                )
            elif PATH_CONFIG.target_benchmark == "B3":
                trend = float((breakdown.get("trend_accuracy") or {}).get("score", 0.0) or 0.0)
                logic = float((breakdown.get("logic") or {}).get("score", 0.0) or 0.0)
                pred_002 = float((breakdown.get("pred_002_accuracy") or {}).get("score", 0.0) or 0.0)
                if "width_factor" in profile:
                    details = ((breakdown.get("pred_002_accuracy") or {}).get("details") or {})
                    adj = recompute_regression_score_from_details(
                        details,
                        profile["width_factor"],
                        profile["width_alpha"],
                        point_weight=profile.get("point_weight", 0.6),
                        interval_weight=profile.get("interval_weight", 0.4),
                    )
                    if adj is not None:
                        pred_002 = adj
                final_score = (
                    SCORING_CONFIG.b3_trend_weight * trend
                    + SCORING_CONFIG.b3_pred_002_weight * pred_002
                    + SCORING_CONFIG.b3_logic_weight * logic
                )

            if final_score is None:
                continue
            rows.append({
                "profile": profile["name"],
                "mode": mode,
                "model_name": model_name,
                "dataset_name": dataset_profile["dataset_name"],
                "scale_bucket": dataset_profile["scale_bucket"],
                "accuracy_score": round(float(acc), 4) if PATH_CONFIG.target_benchmark == "B1" else (
                    round(float(avg_pred), 4) if PATH_CONFIG.target_benchmark == "B2" else round(float(pred_002), 4)
                ),
                "logic_score": round(float(logic), 4),
                "decision_score": round(float(decision), 4) if PATH_CONFIG.target_benchmark == "B2" else None,
                "trend_score": round(float(trend), 4) if PATH_CONFIG.target_benchmark == "B3" else None,
                "final_score": round(float(final_score), 4),
            })

    if not rows:
        return
    report_dir = os.path.join(PATH_CONFIG.inference_root, "_analysis")
    os.makedirs(report_dir, exist_ok=True)
    raw_df = pd.DataFrame(rows)
    suffix = "_full" if full_grid else ""
    raw_df.to_csv(os.path.join(report_dir, f"metric_sensitivity_records_{PATH_CONFIG.target_benchmark}{suffix}.csv"), index=False)
    summary_df = summarize_records_df(raw_df, ["profile", "mode", "model_name"])
    summary_df.to_csv(os.path.join(report_dir, f"metric_sensitivity_summary_{PATH_CONFIG.target_benchmark}{suffix}.csv"), index=False)
    if "scale_bucket" in raw_df.columns:
        scale_df = summarize_records_df(raw_df, ["profile", "mode", "model_name", "scale_bucket"])
        scale_df.to_csv(os.path.join(report_dir, f"metric_sensitivity_by_scale_{PATH_CONFIG.target_benchmark}{suffix}.csv"), index=False)
    if EXPORT_METRIC_RANK_STABILITY or full_grid:
        export_metric_rank_stability(raw_df, report_dir, PATH_CONFIG.target_benchmark, suffix=suffix)
    logger.info(f"已导出指标敏感性结果: {report_dir}")


def export_judge_artifacts() -> None:
    report_dir = os.path.join(PATH_CONFIG.inference_root, "_analysis")
    os.makedirs(report_dir, exist_ok=True)

    prompt_examples = {
        "b1_regression_no_tool": PromptBuilder.build_regression_prompt(
            query="<QUERY>",
            gt_str="<GT>",
            response="<MODEL_RESPONSE>",
            history_snippet="",
            dataset_meta_str="<DATASET_METADATA>",
            mode="no_tool",
        ),
        "b1_classification_with_tool": PromptBuilder.build_classification_prompt(
            query="<QUERY>",
            gt_str="<GT>",
            response="<MODEL_RESPONSE>",
            history_snippet="",
            dataset_meta_str="<DATASET_METADATA>",
            target_col="<TARGET_COLUMN>",
            valid_cats=["A", "B", "C"],
            mode="with_tool",
        ),
        "b2_choice_no_tool": PromptBuilder.build_choice_prompt(
            query="<QUERY>",
            scenarios_info=[{"scenario_id": "001", "features": {"target": "<VALUE_1>"}}, {"scenario_id": "002", "features": {"target": "<VALUE_2>"}}],
            gt_decision="<WINNER_ID>",
            response="<MODEL_RESPONSE>",
            history_snippet="",
            dataset_meta_str="<DATASET_METADATA>",
            target_col="<TARGET_COLUMN>",
            valid_cats=["yes", "no"],
            mode="no_tool",
        ),
        "b3_whatif_no_tool": PromptBuilder.build_whatif_prompt(
            query="<QUERY>",
            scenarios_info=[{"scenario_id": "001", "features": {"target": "<BASELINE>"}}, {"scenario_id": "002", "features": {"target": "<COUNTERFACTUAL>"}}],
            gt_trend="<TREND>",
            response="<MODEL_RESPONSE>",
            history_snippet="",
            dataset_meta_str="<DATASET_METADATA>",
            target_col="<TARGET_COLUMN>",
            valid_cats=["up", "down"],
            mode="no_tool",
        ),
    }

    artifact = {
        "judge_model_id": JUDGE_MODEL_ID,
        "base_url": BASE_URL,
        "workers": MAX_WORKERS,
        "benchmark": PATH_CONFIG.target_benchmark,
        "mode_filter": PATH_CONFIG.target_mode,
        "scoring_config": {
            "accuracy_weight": SCORING_CONFIG.accuracy_weight,
            "logic_weight": SCORING_CONFIG.logic_weight,
            "b2_decision_weight": SCORING_CONFIG.b2_decision_weight,
            "b2_avg_pred_weight": SCORING_CONFIG.b2_avg_pred_weight,
            "b2_logic_weight": SCORING_CONFIG.b2_logic_weight,
            "b3_trend_weight": SCORING_CONFIG.b3_trend_weight,
            "b3_pred_002_weight": SCORING_CONFIG.b3_pred_002_weight,
            "b3_logic_weight": SCORING_CONFIG.b3_logic_weight,
            "overlap_penalty_width_factor": SCORING_CONFIG.overlap_penalty_width_factor,
            "width_penalty_alpha": SCORING_CONFIG.width_penalty_alpha,
            "numeric_tolerance": SCORING_CONFIG.numeric_tolerance,
            "string_similarity_threshold": SCORING_CONFIG.string_similarity_threshold,
        },
        "prompt_templates": prompt_examples,
    }
    safe_json_dump(artifact, os.path.join(report_dir, f"judge_artifacts_{PATH_CONFIG.target_benchmark}.json"))
    logger.info(f"已导出 Judge 配置与模板: {report_dir}")

# ================= 评分计算 =================
class MetricCalculator:
    """评分计算器"""
    
    @staticmethod
    def calc_overlap(pred_min: float, pred_max: float, 
                    gt_min: float, gt_max: float) -> float:
        """计算区间IoU"""
        inter_min = max(pred_min, gt_min)
        inter_max = min(pred_max, gt_max)
        inter_len = max(0, inter_max - inter_min)
        
        union_len = (pred_max - pred_min) + (gt_max - gt_min) - inter_len
        
        if union_len <= 1e-9:
            return 1.0 if inter_len > union_len - 1e-9 else 0.0
        
        return inter_len / union_len
    
    @staticmethod
    def calc_regression(extracted: Dict, gt_value: Any, 
                        dataset_stat: Optional[Dict] = None) -> Tuple[float, Dict]:
        """计算回归任务得分 (修复：防止 float(None) 报错)"""
        try:
            gt_value = float(gt_value)
            pred_val = extracted.get('predicted_value')
            pred_interval = extracted.get('predicted_interval')
            
            # 确定GT范围
            dataset_std = dataset_stat["std"]
            if dataset_std and dataset_std > 1e-9:
                half_width_gt = 0.5 * dataset_std
            else:
                half_width_gt = max(abs(gt_value) * 0.1, 1e-9)
            
            gt_min = gt_value - half_width_gt
            gt_max = gt_value + half_width_gt
            
            # 确定预测范围和点值
            final_pred_point = None
            pred_min, pred_max = None, None
            
            # [关键修改] 安全处理 Interval 中的 None
            valid_interval = False
            if pred_interval and isinstance(pred_interval, list) and len(pred_interval) == 2:
                v1, v2 = pred_interval[0], pred_interval[1]
                
                # 处理半开区间的情况，例如 [300000, null] -> 视为 300000
                if v1 is not None and v2 is None:
                    v2 = v1
                elif v1 is None and v2 is not None:
                    v1 = v2
                
                # 只有当两个值都非 None 时才进行 float 转换
                if v1 is not None and v2 is not None:
                    try:
                        pred_min, pred_max = float(v1), float(v2)
                        final_pred_point = (pred_min + pred_max) / 2.0
                        valid_interval = True
                    except (ValueError, TypeError):
                        pass # 转换失败则回退到 pred_val
            
            # 如果 Interval 无效，尝试使用 predicted_value
            if not valid_interval:
                if pred_val is not None:
                    try:
                        final_pred_point = float(pred_val)
                        pred_min = final_pred_point - half_width_gt
                        pred_max = final_pred_point + half_width_gt
                    except (ValueError, TypeError):
                        return 0.0, {"error": f"无法转换 predicted_value: {pred_val}"}
                else:
                    return 0.0, {"error": "未提取到有效预测值或区间"}
            
        
            # ---------------------------------------------------------
            # 使用 NMAE (Normalized Mean Absolute Error)
            # ---------------------------------------------------------
            
            # 1. 获取数据范围 (Range)
            # if dataset_stat and "q05" in dataset_stat and "q95" in dataset_stat:
            #     d_min = dataset_stat["q05"]
            #     d_max = dataset_stat["q95"]
            #     data_range = d_max - d_min

            if dataset_stat and "min" in dataset_stat and "max" in dataset_stat:
                d_min = dataset_stat["min"]
                d_max = dataset_stat["max"]
                data_range = d_max - d_min
            else:
                # 兜底：如果没有统计数据，回退到使用 GT 的绝对值作为分母 (类似 MAPE)
                # 这种情况很少见，因为你已经有 StatsCache 了
                print("Error: Can't find dataset_stat")
                data_range = 0.0

            abs_error = abs(final_pred_point - gt_value)
            
            # 2. 计算归一化误差
            if data_range > 1e-9:
                # 正常情况：除以范围
                nmae = abs_error / data_range
            else:
                # 特殊情况：范围为0 (常量列) 或 无统计数据
                # 回退策略：如果误差很小(绝对匹配)给满分，否则给0分
                # 或者使用相对误差: abs_error / (abs(gt) + epsilon)
                denom = abs(gt_value) if abs(gt_value) > 1e-9 else 1.0
                nmae = abs_error / denom

            # 3. 转换为分数 (0 ~ 1)
            # 线性映射：NMAE=0 -> 100分; NMAE=0.1 -> 90分; NMAE>=1.0 -> 0分
            score_accuracy = max(0.0, 1.0 - nmae)
            
            # 计算重叠分数
            score_overlap = MetricCalculator.calc_overlap(pred_min, pred_max, gt_min, gt_max)
            
            # 宽度惩罚
            width_penalty_multiplier = 1.0
            pred_width = pred_max - pred_min

            if dataset_std and dataset_std > 1e-9:
                half_width_gt = 0.5 * dataset_std
            else:
                # 兜底：如果 GT 是 0，给一个微小的范围
                half_width_gt = max(abs(gt_value) * 0.1, 1e-6)
            
            gt_min = gt_value - half_width_gt
            gt_max = gt_value + half_width_gt
            gt_width = gt_max - gt_min
            
            # 防止除以零
            safe_gt_width = gt_width if gt_width > 1e-9 else 1e-9
            width_ratio = pred_width / safe_gt_width
            
            threshold = SCORING_CONFIG.overlap_penalty_width_factor
            alpha = SCORING_CONFIG.width_penalty_alpha # 指数衰减系数
            
            if width_ratio > threshold:
                # 只有当宽度超过阈值时才开始惩罚
                # 使用指数衰减：e^(-alpha * 超出量)
                excess = width_ratio - threshold
                width_penalty_multiplier = math.exp(-alpha * excess)
                
                # 限制最小乘数为 0 (虽然 exp 永远 > 0，但为了数值稳定性)
                width_penalty_multiplier = max(0.0, width_penalty_multiplier)

            # 综合得分 = 基础分 * 惩罚乘数
            base_score = 0.6 * score_accuracy + 0.4 * score_overlap
            final_score = base_score * width_penalty_multiplier
            
            return round(max(0.0, min(1.0, final_score)), 4), {
                "gt_value": gt_value, # [新增] 保存原始 GT 值
                "nmae_score": round(score_accuracy, 4),
                "overlap_score": round(score_overlap, 4),
                "pred_point": round(final_pred_point, 4),
                "pred_range": [round(pred_min, 2), round(pred_max, 2)],
                "gt_range": [round(gt_min, 2), round(gt_max, 2)],
                "penalty": round(width_penalty_multiplier, 4)
            }
            
        except Exception as e:
            logger.error(f"回归评分失败: {e}")
            return 0.0, {"error": str(e)}
    
    @staticmethod
    def calc_classification(extracted: Dict, gt_value: Any, 
                          valid_cats: Optional[List] = None) -> Tuple[float, Dict]:
        """计算分类任务得分"""
        try:
            raw_pred = extracted.get('predicted_category')
            if raw_pred is None:
                return 0.0, {"valid": False, "note": "预测为None"}
            
            norm_pred = normalize_text(raw_pred)
            norm_gt = normalize_text(gt_value)
            
            # 精确匹配
            hit = 1.0 if norm_pred == norm_gt else 0.0
            
            # 合法性检查
            is_valid = True
            if valid_cats:
                norm_valid = [normalize_text(str(c)) for c in valid_cats]
                if norm_pred not in norm_valid:
                    is_valid = False
            
            final_score = hit if is_valid else 0.0
            
            return float(final_score), {
                "match": hit == 1.0,
                "pred": norm_pred,
                "gt": norm_gt, # [确认] 包含 GT
                "raw_gt": gt_value, # [新增] 原始 GT
                "valid": is_valid
            }
            
        except Exception as e:
            logger.error(f"分类评分失败: {e}")
            return 0.0, {"error": str(e)}

# ================= Prompts =================
class PromptBuilder:
    """Prompt构建器 (已整合公共逻辑与新版评分标准)"""

    # ================= 公共组件定义 =================

    # 1. 逻辑缺陷检查清单 (所有模式通用)
    COMMON_FLAWS_CHECKLIST = """
    **STEP 1: FLAW DETECTION (CHECK FOR THE "SIX SINS")**
    1. **Self-Contradiction**: Text conflicts with prediction or ignores cited features.
    2. **Tautology**: Circular logic (explaining the result with the result itself).
    3. **Repetitive**: Restating the same point without new info.
    4. **Vacuous Fluff**: Empty fillers ("As an AI...", "Complex analysis") with no substance.
    5. **False Causality**: Linking irrelevant inputs (e.g., IDs) to outputs.
    6. **Over-Hedging**: Refusing to conclude ("Could be A or B").
    """

    # 2. 符号解释规则
    SYMBOL_RULE = """
    **SYMBOL INTERPRETATION RULE**:
    - **Context determines Sign**: Symbols like `−` (unicode minus), `-`, or `~` can be ambiguous.
    - If the text uses `−$912` to mean "**slightly less than $912**" (approximate), you MUST extract **912** (Positive).
    """

    # 3. 反幻觉示例 (Few-Shot)
    ANTI_HALLUCINATION_EXAMPLES = """
    ### 🛑 ANTI-HALLUCINATION EXAMPLES (READ CAREFULLY)
    
    **Scenario A (Rounding Error):**
    - Text: "The growth was approximately 1.4%."
    - ❌ WRONG Extraction: `predicted_interval`: [1, 2] (Reason: The integers "1" and "2" are NOT in the text.)
    - ✅ CORRECT Extraction: `predicted_value`: 1.4 (or null if you only want intervals)
    
    **Scenario B (Summarization Error):**
    - Text: "Values ranged from 1.3 in the south to 1.8 in the north."
    - ❌ WRONG Extraction: `predicted_interval`: [1, 2] (Reason: Do not broaden the range to integers.)
    - ✅ CORRECT Extraction: `predicted_interval`: [1.3, 1.8] (Exact numbers found.)
    
    **Scenario C (Implied Range Error):**
    - Text: "It is likely above 5."
    - ❌ WRONG Extraction: `predicted_interval`: [5, 10] (Reason: "10" is hallucinated.)
    - ✅ CORRECT Extraction: `predicted_value`: null, `predicted_interval`: null (Unless "10" is explicitly upper bound.)

    **Scenario D (Label Rephrasing - FATAL):**
    - Text: "Estimated annual charges: $9,500"
    - ❌ WRONG Quote: "Estimated cost: $9,500" (Reason: "Estimated cost" is NOT in the text. Do not change words.)
    - ✅ CORRECT Quote: "Estimated annual charges: $9,500" (Copy exactly.)
    - Text: "Ah! Here's a 58-year-old non-smoking woman, 0 children, BMI 41.91, Southeast: → Charges: $24,227.34"
    - ❌ WRONG Quote:  "A 58-year-old non-smoking woman in the Southeast with BMI 41.91 had charges of $24,227.34" (Reason: Summarizing in your own words is prohibited.)
    - ✅ CORRECT Quote: "a 58-year-old non-smoking woman, 0 children, BMI 41.91, Southeast: → Charges: $24,227.34" (Copy exactly.)

    **Scenario E (Ordinal/Synonym Replacement - FATAL):**
    - Text: "The first set is likely male."
    - ❌ WRONG Quote: "Set 1 is likely male." (Reason: Do not change "first" to "1". Do not change words.)
    - ✅ CORRECT Quote: "The first set is likely male." (Copy exactly.)
    """

    # 4. 缺失预测处理规则
    MISSING_PREDICTION_RULE = """
    **SILENCE = NULL (DO NOT INFER)**:
    - If the model discusses a Scenario (e.g., analyzes its risk) but **DOES NOT explicitly state** the value for the target column, you MUST set `predicted_value`/`predicted_category` to `null`.
    - **Example**: If text says "Scenario 1 is Class A, Scenario 2 is bad", extract "Class A" for S1, but `null` for S2.
    - **STRICT PROHIBITION**: Do not copy the prediction from Scenario 1 to Scenario 2 unless the text explicitly says "Scenario 2 is the same".
    """

    # 5. 比较处理规则
    COMPARISON_RULE = """
    **HANDLING COMPARISONS (CRITICAL)**:
    - If the text says "Option A is better than Option B":
      - For Option A: Quote "Option A is better than Option B".
      - For Option B: **ALSO Quote "Option A is better than Option B"**.
    - **DO NOT MODIFY THE TEXT**: Never swap names/IDs to make the sentence fit. 
    - **Example**: If text says "ID 37651 has higher risk than ID 10436", do NOT write "ID 10436 has higher risk" as a quote. Quote the original sentence exactly.
    """

    # ================= 动态逻辑生成 =================

    @staticmethod
    def get_logic_criteria(mode: str = "no_tool") -> str:
        """根据模式返回对应的逻辑评分标准 (简化版 + 缺陷清单)"""
        if mode == "with_tool" or "aide" in mode:
            return f"""
    ### 3. LOGIC ASSESSMENT (0-5 SCALE) - MODE: WITH TOOL
    *Note: You cannot see the code. Judge based ONLY on the text description of the execution.*
    
    {PromptBuilder.COMMON_FLAWS_CHECKLIST}

    **STEP 2: SCORING RUBRIC (Concise)**
    **0-1 (Fatal)**: Contains Self-Contradiction, False Causality, or Hallucinated Method.
    **2 (Weak)**: Black Box (result only, no method), Vacuous Fluff, or Repetitive.
    **3 (Pass)**: Transparent Method. Clearly states *what* operation was performed. No flaws.
    **4 (Strong)**: Data Interpretation. Explains *what the tool output implies* beyond just stating the number.
    **5 (Expert)**: Contextual Synthesis. Connects tool results with domain knowledge or identifies patterns.
    """
        else:
            return f"""
    ### 3. LOGIC ASSESSMENT (0-5 SCALE) - MODE: NO TOOL
    *Note: The model must rely on reasoning from the provided features.*
    
    {PromptBuilder.COMMON_FLAWS_CHECKLIST}

    **STEP 2: SCORING RUBRIC (Concise)**
    **0-1 (Fatal)**: Contains Self-Contradiction, False Causality, or Over-Hedging.
    **2 (Weak)**: Tautology, Vacuous Fluff, or Repetitive.
    **3 (Pass)**: Basic Consistency. Logical direction is correct. No flaws.
    **4 (Strong)**: Evidence-Based. Explicitly cites/compares specific input values.
    **5 (Expert)**: Nuanced Synthesis. Weighs conflicting features or captures non-linear trends.
    """

    # ================= 具体的 Prompt 构建方法 =================

    @staticmethod
    def build_regression_prompt(query: str, gt_str: str, response: str, 
                               history_snippet: str, dataset_meta_str: str,
                               mode: str = "no_tool") -> str:
        
        logic_block = PromptBuilder.get_logic_criteria(mode)
        
        return f"""
You are an expert evaluator for a tabular data prediction task (REGRESSION).

Input Data:
[Query]: 
{query}

[Model Response]: 
{response}

[Ground Truth]: 
{gt_str}

[Dataset Metadata]:
{dataset_meta_str}

---
### YOUR TASKS

1. **Prediction Extraction (STRICT)**: 
   - Extract the final numerical prediction or interval.
   - **CRITICAL RULE FOR VAGUE NUMBERS**: 
     - If the text says "2 million+", "over 500k", or "approx 10%", you must extract the **visible number** (e.g., 2000000, 500000, 10).
     - **DO NOT** make up a precise number to represent the "+" (e.g., do NOT convert "2 million+" to 2.4 million). 
     - **Unless** the precise number (2.4 million) is explicitly stated in another part of the text.
   - **CRITICAL RULE FOR INTERVALS**: Do NOT narrow down or calculate. Extract the **EXACT boundaries** mentioned.
     - Text: "likely between 0 and 3 years" -> Prediction: [0, 3]
   - If the prediction value does not exist, set as null.
   {PromptBuilder.SYMBOL_RULE}

2. **Proof Extraction (CRITICAL)**:
   - You MUST copy the `proof_quote` VERBATIM from the [Model Response].
   - **DATA INTEGRITY CHECK**: 
     - The numbers in your `predicted_interval` MUST be **visibly identical** to the numbers in this quote.
     - If you predicted `2400000`, your quote MUST contain "2.4 million" (or similar).
     - If your quote only says "$2 million+", you MUST adjust your prediction to `2000000`.
   - **For Intervals**: Your quote MUST contain the text for **BOTH** the lower and upper bounds.
   - **DO NOT** add property names, keys, or prefixes.
   {PromptBuilder.ANTI_HALLUCINATION_EXAMPLES}
   
{logic_block}

---
### OUTPUT JSON FORMAT
{{
  "prediction_payload": {{
    "predicted_value": number or null,
    "predicted_interval": [min, max] or null,
    "proof_quote_value": "exact substring from response containing the value. Don't add any other words",
    "proof_quote_interval": "exact substring from response containing the interval. Don't add any other words"
  }},
  "logic_assessment": {{
    "logic_score_raw": 0-5,
    "detected_flaws": ["List strings from STEP 1 (e.g. 'Self-Contradiction') or empty []"],
    "reasoning": "Brief justification."
  }}
}}

Output RAW JSON only.
"""
    
    @staticmethod
    def build_classification_prompt(query: str, gt_str: str, response: str,
                                   history_snippet: str, dataset_meta_str: str,
                                   target_col: str, valid_cats: List,
                                   mode: str = "no_tool") -> str:
        
        valid_list_str = str(valid_cats) if valid_cats else "Not provided"
        logic_block = PromptBuilder.get_logic_criteria(mode)

        return f"""
You are an expert evaluator for a tabular data prediction task (CLASSIFICATION).

Input Data:
[Query]: 
{query}

[Model Response]: 
{response}

[Ground Truth]: 
{gt_str}

[Dataset Metadata]:
{dataset_meta_str}

### CRITICAL CONSTRAINT
Target column: **'{target_col}'**
VALID CATEGORIES: **{valid_list_str}**

The predicted_category MUST be EXACTLY one of these values.

---
### YOUR TASKS

1. **Prediction Extraction**: Extract the normalized category string. If the prediction category does not exit, just set as null.

2. **Proof Extraction (CRITICAL)**:
   - You MUST copy the `proof_quote` VERBATIM from the [Model Response].
   - **DO NOT** add property names, keys, or prefixes like "status:", "prediction:", or "result:". 
   - **DO NOT** rephrase or summarize.
   - Example: 
     - Wrong Quote: "Status: Canceled" (If "Status:" is not in text)
     - Correct Quote: "canceled"

{logic_block}

---
### OUTPUT JSON FORMAT
{{
  "prediction_payload": {{
    "predicted_category": "string" or null,
    "proof_quote": "exact substring from response. Don't add any other words"
  }},
  "logic_assessment": {{
    "logic_score_raw": 0-5,
    "detected_flaws": ["List strings from STEP 1 (e.g. 'Self-Contradiction') or empty []"],
    "reasoning": "Brief justification."
  }}
}}

Output RAW JSON only.
"""
    
    @staticmethod
    def build_choice_prompt(query: str, scenarios_info: List[Dict], gt_decision: str, 
                            response: str, dataset_meta_str: str, 
                            target_col: str, task_type: str,
                            valid_cats: Optional[List] = None,
                            mode: str = "no_tool") -> str:
        
        # 1. 构建场景描述
        scenarios_desc = []
        for item in scenarios_info:
            sid = item.get("scenario_id")
            feats = item.get("features", {})
            desc_tokens = [f"scenario_id: {sid}"]
            for k in feats:
                if k == target_col: continue
                desc_tokens.append(f"{k}: {feats[k]}")
            scenarios_desc.append(" | ".join(desc_tokens))
        scenarios_block = "\n".join(scenarios_desc)

        # 2. 获取 Logic 标准
        logic_block = PromptBuilder.get_logic_criteria(mode)

        # 3. 根据任务类型构建特定的提取指令和JSON模板
        if task_type == "classification":
            valid_list_str = str(valid_cats) if valid_cats else "Not provided"
            extraction_instruction = f"""
    - **Task Type**: CLASSIFICATION (Category Extraction)
    - **Valid Categories**: {valid_list_str}
    - For EACH Scenario ID, extract the predicted category for '{target_col}'.
    1.1 **Prediction Extraction**: Extract the normalized category string. If the prediction category does not exit, just set as null.

    1.2 **Proof Extraction (CRITICAL)**:
    - You MUST copy the `proof_quote` VERBATIM from the [Model Response].
    - **DO NOT** add property names, keys, or prefixes like "status:", "prediction:", or "result:". 
    - **DO NOT** rephrase or summarize.
            """
            json_field_template = """
      "predicted_category": "string_category" or null,
      "proof_quote": "MANDATORY if predicted_category exists"
            """
        else: # regression
            extraction_instruction = f"""
    - **Task Type**: REGRESSION (Numeric Extraction)
    - For EACH Scenario ID, extract the predicted numerical value and interval.
    1.1 **Prediction Extraction (STRICT)**: 
    - Extract the final numerical prediction or interval.
    - **CRITICAL RULE FOR VAGUE NUMBERS**: 
        - If the text says "2 million+", "over 500k", or "approx 10%", you must extract the **visible number** (e.g., 2000000, 500000, 10).
    - **CRITICAL RULE FOR INTERVALS**: Extract the **EXACT boundaries** mentioned.
    - If the prediction value does not exist, set as null.

    1.2 **Proof Extraction (CRITICAL)**:
    - You MUST copy the `proof_quote` VERBATIM from the [Model Response].
    - **DATA INTEGRITY CHECK**: The numbers in your `predicted_interval` MUST be **visibly identical** to the numbers in this quote.
            """
            json_field_template = """
      "predicted_value": number or null,
      "predicted_interval": [min, max] or null,
      "proof_quote_value": "MANDATORY string if value exists, otherwise null",
      "proof_quote_interval": "MANDATORY string if interval exists, otherwise null"
            """
        
        return f"""
You are an expert evaluator for a tabular Choice/Ranking task.

Input Data:
[Query]: 
{query}

[Scenarios Context (ID Mapping)]:
{scenarios_block}

[Dataset Metadata]:
{dataset_meta_str}

[Model Response]: 
{response}

[Ground Truth Info]:
Winner ID: {gt_decision}
Target Column: '{target_col}'

---
### YOUR TASKS

1. **Extract Predictions per Scenario**:
{extraction_instruction}
{PromptBuilder.MISSING_PREDICTION_RULE}
{PromptBuilder.ANTI_HALLUCINATION_EXAMPLES}
{PromptBuilder.SYMBOL_RULE}
{PromptBuilder.COMPARISON_RULE}
   - Map the model's textual description back to the Scenario IDs provided above.
   - If no value/category is mentioned for a specific ID, set it to null.

2. **Extract Final Decision**:
   - Identify which Scenario ID the model chose as the best/winner.
   - If the model suggests multiple or none, extract null.

{logic_block}

---
### OUTPUT JSON FORMAT
{{
  "scenarios_extraction": {{
    "001": {{ {json_field_template} }},
    "002": {{ {json_field_template} }},
    ... (one key for each ID in Context)
  }},
  "final_decision_extraction": {{
    "predicted_winner_id": "string_id" or null,
    "proof_quote": "exact substring supporting the decision"
  }},
  "logic_assessment": {{
    "logic_score_raw": 0-5,
    "detected_flaws": ["List strings from STEP 1 or empty []"],
    "reasoning": "Brief justification."
  }}
}}

Output RAW JSON only.
"""

    @staticmethod
    def build_whatif_prompt(query: str, scenarios_info: List[Dict], gt_trend: str, 
                            response: str, dataset_meta_str: str, 
                            target_col: str, task_type: str,
                            valid_cats: Optional[List] = None,
                            mode: str = "no_tool") -> str:
        
        # 1. 构建场景描述
        scenarios_desc = []
        for item in scenarios_info:
            sid = item.get("scenario_id")
            feats = item.get("features", {})
            desc_tokens = [f"Scenario ID: {sid}"]
            for k in feats:
                if k == target_col: continue
                desc_tokens.append(f"{k}: {feats[k]}")
            scenarios_desc.append(" | ".join(desc_tokens))
        scenarios_block = "\n".join(scenarios_desc)

        # 2. 获取 Logic 标准
        logic_block = PromptBuilder.get_logic_criteria(mode)

        # 3. 根据任务类型构建特定的提取指令和JSON模板
        if task_type == "classification":
            valid_list_str = str(valid_cats) if valid_cats else "Not provided"
            allowed_trends = ["same", "change"]
            extraction_instruction = f"""
    - **Task Type**: CLASSIFICATION
    - **Valid Categories**: {valid_list_str}
    - Extract the predicted category for Scenario 002 (the "modified" or "what-if" scenario).
   1.1 **Prediction Extraction**: Extract the normalized category string. If the prediction category does not exit, just set as null.

    1.2 **Proof Extraction (CRITICAL)**:
    - You MUST copy the `proof_quote` VERBATIM from the [Model Response].
    - **DO NOT** add property names, keys, or prefixes like "status:", "prediction:", or "result:". 
    - **DO NOT** rephrase or summarize.
            """
            json_pred_template = """
    "predicted_category": "string" or null,
    "proof_quote": "exact substring or null"
            """
        else: # regression
            allowed_trends = ["lower", "higher", "same"]
            extraction_instruction = f"""
    - **Task Type**: REGRESSION
    - Extract the predicted value AND interval for Scenario 002.
    1.1 **Prediction Extraction (STRICT)**: 
    - Extract the final numerical prediction or interval.
    - **CRITICAL RULE FOR VAGUE NUMBERS**: 
        - If the text says "2 million+", "over 500k", or "approx 10%", you must extract the **visible number** (e.g., 2000000, 500000, 10).
    - **CRITICAL RULE FOR INTERVALS**: Extract the **EXACT boundaries** mentioned.
    - If the prediction value does not exist, set as null.

    1.2 **Proof Extraction (CRITICAL)**:
    - You MUST copy the `proof_quote` VERBATIM from the [Model Response].
    - **DATA INTEGRITY CHECK**: The numbers in your `predicted_interval` MUST be **visibly identical** to the numbers in this quote.
            """
            json_pred_template = """
    "predicted_value": number or null,
    "predicted_interval": [min, max] or null,
    "proof_quote_value": "MANDATORY string if value exists",
    "proof_quote_interval": "MANDATORY string if interval exists"
            """

        narrative_prediction_rule = f"""
    **HANDLING NARRATIVE SHIFTS & CONTINUITY (CRITICAL)**:
    - In What-If tasks, the model describes the outcome of Scenario 002 (compared to Scenario 001).
    - **RULE 1 (Explicit Change)**: 
      - If text says "elevate to Y", "increase into the Y range", "result in Y", or "drop to Y", then **Y is the prediction**.
      
    - **RULE 2 (Continuity/Stability)**: 
      - If text says "**stay at Y**", "**remain Y**", "**maintain the rating of Y**", or "**likely stay at Y**", then **Y is the prediction** for Scenario 002.
        """

        return f"""
You are an expert evaluator for a tabular What-If analysis task.

Input Data:
[Query]: 
{query}

[Scenarios Context]:
{scenarios_block}

[Dataset Metadata]:
{dataset_meta_str}

[Model Response]: 
{response}

[Ground Truth Info]:
Actual Trend: {gt_trend}
Target Column: '{target_col}'

---
### YOUR TASKS

1. **Extract Scenario 002 Prediction**:
    - Focus on Scenario 002 (the hypothetical/modified case).
{extraction_instruction}
{narrative_prediction_rule}
{PromptBuilder.MISSING_PREDICTION_RULE}
{PromptBuilder.ANTI_HALLUCINATION_EXAMPLES}
{PromptBuilder.SYMBOL_RULE}
{PromptBuilder.COMPARISON_RULE}

2. **Extract Trend Conclusion**:
    - Determine the model's conclusion on how the target changes from 001 to 002.
    - VALID OPTIONS: {allowed_trends}
    - Extract the `proof_quote` that supports this trend conclusion.

{logic_block}

---
### OUTPUT JSON FORMAT
{{
  "scenario_002_extraction": {{
    {json_pred_template}
  }},
  "trend_extraction": {{
    "predicted_trend": "{'/'.join(allowed_trends)}" or null,
    "proof_quote": "exact substring supporting the trend"
  }},
  "logic_assessment": {{
    "logic_score_raw": 0-5,
    "detected_flaws": ["List strings from STEP 1 or empty []"],
    "reasoning": "Brief justification."
  }}
}}

Output RAW JSON only.
"""
    
# ================= API与流程控制 =================
class AsyncAPIClient:
    """异步 API 客户端 (解决 I/O 阻塞问题)"""
    
    @staticmethod
    async def generate_completion_with_retry(max_retries: int = 5, 
                                             base_delay: float = 1.0, 
                                             **kwargs) -> Any:
        # if "max_tokens" not in kwargs:
        #     kwargs["max_tokens"] = 4096
            
        # 获取当前的 API Key (假设 key_iterator 已经是线程安全的或在此处处理)
        # 在异步环境下，直接获取 next 是安全的，因为 GIL 依然存在，或者使用 asyncio.Lock
        current_api_key = next(key_iterator) 
        client = AsyncOpenAI(api_key=current_api_key, base_url=BASE_URL, timeout=90.0)

        for attempt in range(max_retries):
            try:
                return await client.chat.completions.create(**kwargs)
            
            except (RateLimitError, APIConnectionError) as e:
                wait_time = base_delay * (2 ** attempt)
                logger.warning(f"Async API调用失败 (attempt {attempt+1}): {e}")
                await asyncio.sleep(wait_time) # 非阻塞等待
                
            except (AuthenticationError, APIStatusError) as e:
                logger.error(f"API认证或状态错误: {e}")
                raise
                
            except Exception as e:
                logger.error(f"未知API错误: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(base_delay)
        
        raise Exception("Async API调用失败，达最大重试次数")
    
class APIClient:
    """API客户端"""
    
    @staticmethod
    def generate_completion_with_retry(max_retries: int = 8, 
                                      base_delay: float = 2.0, 
                                      **kwargs) -> Any:
        """带重试的API调用"""
        # if "max_tokens" not in kwargs:
        #     kwargs["max_tokens"] = 4096

        if JUDGE_BACKEND == "gemini":
            prompt = ""
            messages = kwargs.get("messages") or []
            if messages:
                prompt = messages[-1].get("content", "")
            temperature = kwargs.get("temperature", 0.0)
            response_mime_type = "application/json" if kwargs.get("response_format") else "text/plain"

            for attempt in range(max_retries):
                current_api_key = next(key_iterator)
                try:
                    client = genai.Client(api_key=current_api_key)
                    response = client.models.generate_content(
                        model=JUDGE_MODEL_ID,
                        contents=[prompt],
                        config=genai_types.GenerateContentConfig(
                            temperature=temperature,
                            response_mime_type=response_mime_type,
                        ),
                    )
                    text = getattr(response, "text", None)
                    if not text:
                        raise ValueError("Gemini returned empty text response")
                    return _SimpleCompletion(text)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    wait_time = base_delay * (2 ** attempt)
                    logger.warning(f"Gemini Judge API调用失败 (attempt {attempt+1}/{max_retries}): {e}")
                    time.sleep(wait_time)

        for attempt in range(max_retries):
            current_api_key = next(key_iterator)
            client = OpenAI(api_key=current_api_key, base_url=BASE_URL, timeout=90.0)
            
            try:
                return client.chat.completions.create(**kwargs)
            
            except (RateLimitError, APIConnectionError) as e:
                wait_time = base_delay * (2 ** attempt)
                logger.warning(f"API调用失败 (attempt {attempt+1}/{max_retries}): {e}")
                logger.info(f"等待 {wait_time}秒后重试...")
                time.sleep(wait_time)
                
            except (AuthenticationError, APIStatusError) as e:
                logger.error(f"API认证或状态错误: {e}")
                raise
                
            except Exception as e:
                logger.error(f"未知API错误: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(base_delay)
        
        raise Exception(f"API调用失败，已达最大重试次数 {max_retries}")

class LLMJudge:
    """LLM评判器"""

    @staticmethod
    def _validate_entry(response_text: str, 
                        extraction_item: Dict, 
                        task_type: str, 
                        target_stat: Dict, 
                        valid_cats: Optional[List], 
                        attempt: int, 
                        max_attempts: int,
                        context_prefix: str = "") -> Dict:
        """
        [公共验证逻辑] 统一处理 B1, B2, B3 的提取项验证 
        (修复：区间转单点仅用于量级检查，不再回写到 payload，保持原始输出的 null 状态)
        """
        payload = {}
        
        # ---------------------------------------------------------
        # 1. 初步提取与格式化
        # ---------------------------------------------------------
        if task_type == "classification":
            pred_cat = extraction_item.get("predicted_category")
            
            if isinstance(pred_cat, str) and pred_cat.lower().strip() in ["null", "none"]:
                pred_cat = None

            payload = {
                "predicted_category": pred_cat,
                "proof_quote": extraction_item.get("proof_quote")
            }
            
            # 分类有效性检查
            if valid_cats and pred_cat is not None:
                clean_pred = str(pred_cat).strip()
                is_valid_cat = False
                for vc in valid_cats:
                    vc_str = str(vc).strip()
                    if clean_pred == vc_str:
                        is_valid_cat = True; break
                    try:
                        if float(clean_pred) == float(vc):
                            is_valid_cat = True; break
                    except (ValueError, TypeError): pass

                if not is_valid_cat:
                    raise ValueError(f"{context_prefix}Category '{pred_cat}' is not in valid list: {valid_cats}")

        else: # regression
            pred_val = extraction_item.get("predicted_value")
            
            if isinstance(pred_val, str) and pred_val.lower().strip() in ["null", "none"]:
                pred_val = None

            quote_val = extraction_item.get("proof_quote_value") or extraction_item.get("proof_quote")
            quote_int = extraction_item.get("proof_quote_interval")

            # [新增逻辑] 检查 "有引用无值" 的不一致情况
            if pred_val is None and quote_val:
                raise ValueError(f"{context_prefix}Inconsistency detected: proof_quote_value '{quote_val}' is provided, but predicted_value is null. If the text contains a value for this quote, please extract it. If there is no specific value, set the quote to null.")

            # [补充逻辑] 检查 "有区间引用无区间值" (保持一致性)
            pred_int = extraction_item.get("predicted_interval")
            if (pred_int is None or (isinstance(pred_int, list) and not pred_int)) and quote_int:
                 raise ValueError(f"{context_prefix}Inconsistency detected: proof_quote_interval '{quote_int}' is provided, but predicted_interval is null. Please extract the interval numbers or remove the quote.")
            
            payload = {
                "predicted_value": pred_val,
                "predicted_interval": extraction_item.get("predicted_interval"),
                "proof_quote_value": quote_val,
                "proof_quote_interval": quote_int
            }

        # ---------------------------------------------------------
        # 2. 幻觉/引用校验 (针对原始输出进行校验)
        # ---------------------------------------------------------
        is_valid, err_msg = HallucinationVerifier.verify(response_text, payload, task_type, context_prefix=context_prefix)
        
        if not is_valid:
            # if attempt == max_attempts - 1:
            #     logger.warning(f"{context_prefix}达到最大重试次数，校验仍失败 ({err_msg})。将预测值重置为 None 以继续。")
            #     payload["predicted_value"] = None
            #     payload["predicted_interval"] = None
            #     payload["predicted_category"] = None
            #     return payload
            
            raise ValueError(f"Verification Failed: {err_msg}")

        # ---------------------------------------------------------
        # 3. 后处理：量级检查 (使用临时变量，不修改 payload)
        # ---------------------------------------------------------
        if task_type != "classification":
            
            # A. 检查并修正点值
            val_point = payload.get("predicted_value")
            if val_point is not None:
                # 接收三个返回值: valid, error, fixed_val
                is_valid, err, fixed_val = MagnitudeChecker.check_magnitude_and_bounds(val_point, target_stat, response_text)
                
                if not is_valid:
                    if attempt < max_attempts - 1:
                        raise ValueError(f"{context_prefix}{err}")
                    else:
                        logging.warning(f"{context_prefix}量级检查失败但强制接受: {err}")
                
                # [核心] 如果修正后的值存在，更新 payload
                if fixed_val is not None:
                    payload["predicted_value"] = fixed_val
            
            # B. 检查并修正区间端点
            val_interval = payload.get("predicted_interval")
            if val_interval and isinstance(val_interval, list) and len(val_interval) == 2:
                v1, v2 = val_interval[0], val_interval[1]
                new_interval = [v1, v2]
                
                # Check v1
                if v1 is not None:
                    is_valid, err, fixed_v1 = MagnitudeChecker.check_magnitude_and_bounds(v1, target_stat, response_text)
                    if not is_valid and attempt < max_attempts - 1:
                         raise ValueError(f"{context_prefix}Interval Lower: {err}")
                    if fixed_v1 is not None: new_interval[0] = fixed_v1
                
                # Check v2
                if v2 is not None:
                    is_valid, err, fixed_v2 = MagnitudeChecker.check_magnitude_and_bounds(v2, target_stat, response_text)
                    if not is_valid and attempt < max_attempts - 1:
                         raise ValueError(f"{context_prefix}Interval Upper: {err}")
                    if fixed_v2 is not None: new_interval[1] = fixed_v2
                
                payload["predicted_interval"] = new_interval

        return payload

    @staticmethod
    def run_judge(task_type: str, query: str, gt_str: str, response: str,
                  history_snippet: str, dataset_meta_str: str,
                  target_col: Optional[str] = None,
                  valid_cats: Optional[List] = None,
                  stats: Optional[Dict] = None,
                  is_debug: bool = False,
                  subtask_type: str = "single_point",
                  scenarios_info: Optional[List[Dict]] = None,
                  gt_decision: Optional[str] = None,
                  gt_trend: Optional[str] = None
                  ) -> Dict:
        """运行LLM评判"""
        
        max_judge_attempts = 5
        target_stat = stats.get(target_col, {}) if stats and target_col else {}
        last_error_msg = ""
        last_raw_response = ""
        
        for attempt in range(max_judge_attempts):
            
            if subtask_type == "whatif" and scenarios_info:
                # [新增] B3 Prompt
                prompt = PromptBuilder.build_whatif_prompt(
                    query, scenarios_info, gt_trend, response, dataset_meta_str,
                    target_col, task_type, valid_cats
                )
            elif subtask_type == "choice" and scenarios_info:
                prompt = PromptBuilder.build_choice_prompt(
                    query, scenarios_info, gt_decision, response, dataset_meta_str, 
                    target_col, task_type, valid_cats
                )
            elif task_type == "classification":
                prompt = PromptBuilder.build_classification_prompt(
                    query, gt_str, response, history_snippet, dataset_meta_str,
                    target_col, valid_cats
                )
            else:
                prompt = PromptBuilder.build_regression_prompt(
                    query, gt_str, response, history_snippet, dataset_meta_str
                )
            
            # ... (Prompt 错误注入部分保持不变) ...
            if attempt > 0:
                error_instruction = f"""
\n
################################################################################
!!! PREVIOUS ATTEMPT FAILED VALIDATION !!!
The previous output was invalid. 
Error Details: {last_error_msg}

INSTRUCTION: Please fix the error described above. Ensure the JSON format is valid.
################################################################################
"""
                prompt += error_instruction
                if task_type == "classification" and valid_cats:
                    prompt += f"\nENSURE predicted_category is EXACTLY one of: {valid_cats}"
                
                # ================= [新增] 打印重试时的 Prompt =================
                # logger.info(f"\n{'!'*40} RETRY ATTEMPT {attempt} PROMPT {'!'*40}\n{prompt}\n{'!'*100}")
                logger.info(f"\n{'!'*40} PREVIOUS FAILED RESPONSE (Attempt {attempt-1}) {'!'*40}\n{last_raw_response}")
                # ============================================================

            if is_debug:
                log_prefix = f"Attempt {attempt+1} [{subtask_type}]"
                logger.debug(f"\n{'='*50}\nLLM Judge Prompt ({log_prefix}):\n{prompt}\n{'='*50}")

            try:
                completion = APIClient.generate_completion_with_retry(
                    model=JUDGE_MODEL_ID,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                
                raw_content = completion.choices[0].message.content
                last_raw_response = raw_content
                cleaned_content = extract_json_content(raw_content)
                parsed_json = json.loads(cleaned_content)
                
                if is_debug:
                    logger.debug(f"\nLLM Judge Response:\n{json.dumps(parsed_json, indent=2, ensure_ascii=False)}")
                
                # ==========================================
                # 2. 结果验证 (逻辑统一重构)
                # ==========================================

                if subtask_type == "whatif":
                    if "scenario_002_extraction" not in parsed_json:
                        raise ValueError("Missing 'scenario_002_extraction'")
                    if "trend_extraction" not in parsed_json:
                        raise ValueError("Missing 'trend_extraction'")
                    
                    # 2.1 验证 Scenario 002 的提取 (复用 _validate_entry)
                    s002_item = parsed_json["scenario_002_extraction"]
                    if s002_item:
                        # 将标准化后的值回写 (主要是为了 regression 的区间转单点)
                        validated_payload = LLMJudge._validate_entry(
                            response, s002_item, task_type, target_stat, valid_cats,
                            attempt, max_judge_attempts, context_prefix="Scenario ID 002 "
                        )
                        # if task_type == "regression" and validated_payload.get("predicted_value") is not None:
                        #     s002_item["predicted_value"] = validated_payload["predicted_value"]
                        s002_item.update(validated_payload)

                    # 2.2 [修改] 验证 Trend 提取 (现在包含 proof_quote 审查)
                    trend_ext = parsed_json["trend_extraction"]
                    trend_item = {
                        "predicted_category": trend_ext.get("predicted_trend"), # 映射字段名以适配 validate
                        "proof_quote": trend_ext.get("proof_quote")
                    }
                    
                    # 确定 Trend 的合法类别
                    allowed_trends = ["lower", "higher", "same"] if task_type == "regression" else ["change", "same"]
                    
                    # 使用 _validate_entry 进行完整校验 (幻觉检测 + 类别合法性)
                    LLMJudge._validate_entry(
                        response_text=response,
                        extraction_item=trend_item,
                        task_type="classification", # Trend 本质是分类任务
                        target_stat={}, 
                        valid_cats=allowed_trends,
                        attempt=attempt,
                        max_attempts=max_judge_attempts,
                        context_prefix="Trend Extraction"
                    )
                    
                    return parsed_json
                
                # --- 分支 A: B2 (Choice) ---
                if subtask_type == "choice":
                    if "scenarios_extraction" not in parsed_json:
                        raise ValueError("Missing 'scenarios_extraction' field")
                    if "final_decision_extraction" not in parsed_json:
                        raise ValueError("Missing 'final_decision_extraction' field")
                    
                    scenarios_ext = parsed_json["scenarios_extraction"]
                    
                    # 1. 验证每个 Scenario (复用公共函数)
                    for sid, item in scenarios_ext.items():
                        if not item: continue
                        
                        # 调用公共验证函数
                        # 注意：_validate_entry 会返回标准化后的 payload，如果需要可以回写，这里主要用于检查
                        validated_payload = LLMJudge._validate_entry(
                            response_text=response,
                            extraction_item=item,
                            task_type=task_type,
                            target_stat=target_stat,
                            valid_cats=valid_cats,
                            attempt=attempt,
                            max_attempts=max_judge_attempts,
                            context_prefix=f"Scenario ID '{sid}' "
                        )
                        
                        # [可选] 将标准化后的值（如区间转单点）回写到 parsed_json，方便 evaluate 使用
                        # if task_type == "regression" and validated_payload.get("predicted_value") is not None:
                        #     scenarios_ext[sid]["predicted_value"] = validated_payload["predicted_value"]
                        scenarios_ext[sid].update(validated_payload)

                    # 2. 验证最终决策 (视为分类任务)
                    dec_ext = parsed_json["final_decision_extraction"]
                    dec_item = {
                        "predicted_category": dec_ext.get("predicted_winner_id"),
                        "proof_quote": dec_ext.get("proof_quote")
                    }
                    
                    # [优化] 提取所有有效的 Scenario ID
                    valid_scenario_ids = [str(s['scenario_id']) for s in scenarios_info] if scenarios_info else []

                    # 传入 valid_scenario_ids 进行严格检查
                    LLMJudge._validate_entry(
                        response_text=response,
                        extraction_item=dec_item,
                        task_type="classification",
                        target_stat={}, 
                        valid_cats=valid_scenario_ids, # [修改] 传入 ID 列表
                        attempt=attempt,
                        max_attempts=max_judge_attempts,
                        context_prefix="Final Decision "
                    )

                    return parsed_json

                # --- 分支 B: B1 (Single Point) ---
                else:
                    if "prediction_payload" not in parsed_json:
                        raise ValueError("Missing 'prediction_payload' field")
                    
                    payload_raw = parsed_json["prediction_payload"]
                    
                    # 调用公共验证函数
                    validated_payload = LLMJudge._validate_entry(
                        response_text=response,
                        extraction_item=payload_raw,
                        task_type=task_type,
                        target_stat=target_stat,
                        valid_cats=valid_cats,
                        attempt=attempt,
                        max_attempts=max_judge_attempts
                    )
                    
                    # 回写标准化后的 Payload (主要是为了把 Interval 计算出的 predicted_value 写回去)
                    parsed_json["prediction_payload"] = validated_payload
                    
                    return parsed_json
                
            except Exception as e:
                last_error_msg = str(e)
                logger.warning(f"评判尝试 {attempt+1} 失败: {e}")
                
                if attempt == max_judge_attempts - 1:
                    fallback = {"error": f"最终失败: {e}"}
                    if subtask_type == "choice":
                         fallback.update({"scenarios_extraction": {}, "final_decision_extraction": {}})
                    else:
                         fallback.update({"prediction_payload": {"predicted_value": None, "predicted_category": None}})
                    return fallback
                
                time.sleep(1)
        
        return {"error": "未知错误"}

# ================= 主评估流程 =================
def _get_valid_categories(task_type: str, 
                          target_col: str, 
                          real_stats: Dict, 
                          ds_context: Dict) -> Optional[List]:
    """[公共函数] 获取分类任务的有效类别列表"""
    if task_type != "classification":
        return None
        
    target_stat = real_stats.get(target_col, {})
    # 优先顺序: Context显式指定 -> 统计离散值 -> 统计类别
    return (
        ds_context.get("valid_categories") or 
        target_stat.get("discrete_values") or 
        target_stat.get("categories")
    )

def _calculate_score_dispatch(task_type: str, 
                              pred_item: Dict, 
                              gt_val: Any, 
                              valid_cats: Optional[List], 
                              real_stats: Dict, 
                              target_col: str) -> Tuple[float, Dict]:
    """[公共函数] 分发评分计算 (Payload 结构已统一)"""
    
    # 构造 Payload
    temp_payload = {
        "predicted_value": pred_item.get("predicted_value"),
        "predicted_interval": pred_item.get("predicted_interval"),
        "predicted_category": pred_item.get("predicted_category"),
        
        # 此时 pred_item 应该已经包含了详细的 keys (由 _validate_entry 保证或回写)
        "proof_quote_value": pred_item.get("proof_quote_value") or pred_item.get("proof_quote"), 
        "proof_quote_interval": pred_item.get("proof_quote_interval"), # [新增]
        "proof_quote": pred_item.get("proof_quote")
    }

    if task_type == "classification":
        return MetricCalculator.calc_classification(
            temp_payload, gt_val, valid_cats=valid_cats
        )
    else:
        std_val = real_stats.get(target_col, {}).get("std")
        # calc_regression 内部已经支持处理 predicted_interval
        return MetricCalculator.calc_regression(
            temp_payload, gt_val, dataset_stat=real_stats.get(target_col, {})
        )
    
class PreProcessor:
    """响应文本预处理器：压缩长度、去除噪音、智能截断"""
    
    @staticmethod
    def clean_and_compress(text: str, max_chars: int = 15000) -> str:
        if not text:
            return ""
            
        # 1. 移除 Markdown 噪音 (保留内容，去除格式)
        # 移除加粗/斜体 (**word**, *word*)
        # text = text.replace("**", "").replace("__", "")
        
        # 移除 Markdown 标题标记 (### Title -> Title)
        # text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        
        # 移除分割线 (---, ===)
        # text = re.sub(r'[-=]{3,}', ' ', text)
        
        # 移除表格的多余格式字符 (保留内容，但在复杂表格中需谨慎，这里仅把 | 换成空格)
        # text = text.replace("|", " ") 
        
        # 2. 压缩空白字符
        # 将连续的换行符缩减为1个
        text = re.sub(r'\n\s*\n', '\n', text)
        # 将连续的空格缩减为1个 (可选，视情况而定)
        text = re.sub(r'[ \t]+', ' ', text)
        
        text = text.strip()
            
        return text
    
def evaluate_single_file(inference_path: str, 
                          dataset_path: str,
                          model_name: str,
                          benchmark_type: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[str]]:
    """
    评估单个推理文件 (修改版：Skip检查优先于空响应检查)
    返回: (final_score, acc_score, logic_score, decision_score, err_msg)
    err_msg 为 "EMPTY_RESPONSE" 时表示空响应导致0分
    """
    is_debug = (MAX_WORKERS == 1)
    
    try:
        # 1. 基础数据加载
        data = safe_json_load(inference_path)
        if not data: return None, None, None, None, "JSON加载失败"
        
        if not os.path.exists(dataset_path):
             return None, None, None, None, f"数据集路径不存在: {dataset_path}"

        ds_context = DatasetContextLoader.load_dataset_context(dataset_path)
        real_stats = ds_context.get("stats", {})
        prompt_meta = ds_context.get("prompt_meta_str", "")
        ground_truth = data.get("ground_truth", {})
        target_col = ground_truth.get("target_column")
        task_type = ground_truth.get("task_sub_type", ds_context.get("task_type_hint", "regression"))

        valid_cats = _get_valid_categories(task_type, target_col, real_stats, ds_context)

        raw_response = data.get("response", "")
        clean_response = PreProcessor.clean_and_compress(raw_response)

        # 注意：此处不再提前检查 clean_response，而是推迟到各分支 Skip 逻辑之后

        # 初始化返回值变量
        final_score = 0.0
        acc_score = 0.0 
        logic_score = 0.0
        decision_score = None 
        eval_result = {}

        # ==============================================================================
        # BRANCH C: B3 Benchmark (What-If 模式)
        # ==============================================================================
        if benchmark_type == "B3":
            scenarios_info = ground_truth.get("extracted_features", [])
            gt_trend = ground_truth.get("what_if")
            
            # --- Skip Checks (Priority 1) ---
            if len(scenarios_info) < 2: return None, None, None, None, "SKIPPED_B3_MISSING_SCENARIOS"
            
            gt_val_002 = None
            for s in scenarios_info:
                if str(s.get("scenario_id")) == "002":
                    gt_val_002 = s.get("features", {}).get(target_col)
                    break
            
            if gt_val_002 is None: return None, None, None, None, "SKIPPED_B3_NO_GT_002"

            # --- Empty Response Check (Priority 2) ---
            if not clean_response:
                eval_result = {
                    "model_name": model_name, "final_score": 0.0,
                    "task_type": f"B3_{task_type}", "benchmark": benchmark_type,
                    "breakdown": {
                        "trend_accuracy": {"score": 0.0, "pred": None, "gt": gt_trend},
                        "pred_002_accuracy": {"score": 0.0, "details": "Empty Response"},
                        "logic": {"score": 0.0}
                    },
                    "judge_output": {"error": "Empty Response"}
                }
                out_path = get_eval_path(inference_path)
                safe_json_dump(eval_result, out_path, indent=2)
                return 0.0, 0.0, 0.0, 0.0, "EMPTY_RESPONSE"

            # --- LLM Judge ---
            judge_result = LLMJudge.run_judge(
                task_type=task_type, query=data.get("query", ""), gt_str="", 
                response=clean_response, history_snippet="", dataset_meta_str=prompt_meta,
                target_col=target_col, valid_cats=valid_cats, stats=real_stats, is_debug=is_debug,
                subtask_type="whatif", scenarios_info=scenarios_info, gt_trend=gt_trend 
            )

            if "error" in judge_result: return None, None, None, None, judge_result["error"]

            pred_trend = judge_result.get("trend_extraction", {}).get("predicted_trend")
            trend_score = 1.0 if normalize_text(pred_trend) == normalize_text(gt_trend) else 0.0
            
            pred_item_002 = judge_result.get("scenario_002_extraction", {})
            pred_score_002, pred_details_002 = _calculate_score_dispatch(
                task_type, pred_item_002, gt_val_002, valid_cats, real_stats, target_col
            )

            logic_score = float(judge_result.get("logic_assessment", {}).get("logic_score_raw", 0.0)) / 5.0

            final_score = (
                SCORING_CONFIG.b3_trend_weight * trend_score +
                SCORING_CONFIG.b3_pred_002_weight * pred_score_002 +
                SCORING_CONFIG.b3_logic_weight * logic_score
            )

            eval_result = {
                "model_name": model_name, "final_score": round(final_score, 4),
                "task_type": f"B3_{task_type}", "benchmark": benchmark_type,
                "breakdown": {
                    "trend_accuracy": {"score": trend_score, "pred": pred_trend, "gt": gt_trend},
                    "pred_002_accuracy": {
                        "score": round(pred_score_002, 4),
                        "details": pred_details_002
                    },
                    "logic": {"score": round(logic_score, 4)}
                },
                "judge_output": judge_result
            }
            
            out_path = get_eval_path(inference_path)
            safe_json_dump(eval_result, out_path, indent=2)
            
            return (round(final_score, 4), round(pred_score_002, 4), round(logic_score, 4), round(trend_score, 4), None)

        # ==============================================================================
        # BRANCH A: B2 Benchmark (多方案/Choice 模式)
        # ==============================================================================
        if benchmark_type == "B2":
            scenarios_info = ground_truth.get("extracted_features", [])
            gt_decision = ground_truth.get("final_decision", "")
            
            # --- Skip Checks (Priority 1) ---
            if not scenarios_info: return None, None, None, None, "SKIPPED_NO_SCENARIOS_IN_B2"

            # --- Empty Response Check (Priority 2) ---
            if not clean_response:
                eval_result = {
                    "model_name": model_name, "final_score": 0.0,
                    "task_type": f"B2_{task_type}", "benchmark": benchmark_type,
                    "breakdown": {
                        "decision": {"score": 0.0, "pred": None, "gt": gt_decision},
                        "avg_prediction": {"score": 0.0, "count": 0},
                        "scenarios_details": {},
                        "logic": {"score": 0.0}
                    },
                    "judge_output": {"error": "Empty Response"}
                }
                out_path = get_eval_path(inference_path)
                safe_json_dump(eval_result, out_path, indent=2)
                return 0.0, 0.0, 0.0, 0.0, "EMPTY_RESPONSE"

            # --- LLM Judge ---
            judge_result = LLMJudge.run_judge(
                task_type=task_type, query=data.get("query", ""), gt_str="", 
                response=clean_response, history_snippet="", dataset_meta_str=prompt_meta,
                target_col=target_col, valid_cats=valid_cats, stats=real_stats, is_debug=is_debug,
                subtask_type="choice", scenarios_info=scenarios_info, gt_decision=gt_decision
            )

            if "error" in judge_result: return None, None, None, None, judge_result["error"]

            pred_decision_id = judge_result.get("final_decision_extraction", {}).get("predicted_winner_id")
            decision_score = 1.0 if normalize_text(pred_decision_id) == normalize_text(gt_decision) else 0.0
            logic_score = float(judge_result.get("logic_assessment", {}).get("logic_score_raw", 0.0)) / 5.0

            scenarios_extraction = judge_result.get("scenarios_extraction", {})
            acc_scores_list = []
            scenarios_details_map = {} 
            
            for item in scenarios_info:
                sid = item.get("scenario_id")
                gt_val = item.get("features", {}).get(target_col)
                if gt_val is None: continue 
                
                pred_item = scenarios_extraction.get(sid, {})
                s_acc, s_details = _calculate_score_dispatch(
                    task_type, pred_item, gt_val, valid_cats, real_stats, target_col
                )
                acc_scores_list.append(s_acc)
                
                scenarios_details_map[sid] = {
                    "score": s_acc,
                    "gt": gt_val,
                    "pred": pred_item.get("predicted_value") or pred_item.get("predicted_category"),
                    "details": s_details
                }
            
            acc_score = sum(acc_scores_list) / len(acc_scores_list) if acc_scores_list else 0.0

            final_score = (
                SCORING_CONFIG.b2_decision_weight * decision_score +
                SCORING_CONFIG.b2_avg_pred_weight * acc_score +
                SCORING_CONFIG.b2_logic_weight * logic_score
            )

            eval_result = {
                "model_name": model_name, "final_score": round(final_score, 4),
                "task_type": f"B2_{task_type}", "benchmark": benchmark_type,
                "breakdown": {
                    "decision": {"score": decision_score, "pred": pred_decision_id, "gt": gt_decision},
                    "avg_prediction": {"score": round(acc_score, 4), "count": len(acc_scores_list)},
                    "scenarios_details": scenarios_details_map, 
                    "logic": {"score": round(logic_score, 4)}
                },
                "judge_output": judge_result
            }

            out_path = get_eval_path(inference_path)
            safe_json_dump(eval_result, out_path, indent=2)

            # [修复] 之前这里缺失了 Return 语句！
            return round(final_score, 4), round(acc_score, 4), round(logic_score, 4), decision_score, None

        # ==============================================================================
        # BRANCH B: B1 Benchmark (单点/Single 模式)
        # ==============================================================================
        else: 
            extracted_list = ground_truth.get("extracted_features", [])
            gt_val_real = extracted_list[0].get("features", {}).get(target_col) if extracted_list else None
            
            # --- Skip Checks (Priority 1) ---
            if gt_val_real is None: return None, None, None, None, "SKIPPED_NO_GT_IN_B1"

            # --- Empty Response Check (Priority 2) ---
            if not clean_response:
                eval_result = {
                    "model_name": model_name, "final_score": 0.0,
                    "task_type": f"B1_{task_type}", "benchmark": benchmark_type,
                    "breakdown": {
                        "accuracy": {"score": 0.0, "details": "Empty Response"},
                        "logic": {"score": 0.0}
                    },
                    "judge_output": {"error": "Empty Response"}
                }
                out_path = get_eval_path(inference_path)
                safe_json_dump(eval_result, out_path, indent=2)
                return 0.0, 0.0, 0.0, None, "EMPTY_RESPONSE"

            gt_str_prompt = f"Target: {target_col}, GT: {gt_val_real}, Task: {task_type}"

            judge_result = LLMJudge.run_judge(
                task_type=task_type, query=data.get("query", ""), gt_str=gt_str_prompt,
                response=clean_response, history_snippet="", dataset_meta_str=prompt_meta,
                target_col=target_col, valid_cats=valid_cats, stats=real_stats, is_debug=is_debug,
                subtask_type="single_point"
            )
            
            if "error" in judge_result: return None, None, None, None, judge_result["error"]
            
            logic_score = float(judge_result.get("logic_assessment", {}).get("logic_score_raw", 0.0)) / 5.0
            pred_payload = judge_result.get("prediction_payload", {})
            
            s_acc, acc_details = _calculate_score_dispatch(
                task_type, pred_payload, gt_val_real, valid_cats, real_stats, target_col
            )
            acc_score = s_acc

            final_score = (
                SCORING_CONFIG.accuracy_weight * acc_score +
                SCORING_CONFIG.logic_weight * logic_score
            )
            
            eval_result = {
                "model_name": model_name, "final_score": round(final_score, 4),
                "task_type": f"B1_{task_type}", "benchmark": benchmark_type,
                "breakdown": {
                    "accuracy": {"score": round(acc_score, 4), "details": acc_details},
                    "logic": {"score": round(logic_score, 4)}
                },
                "judge_output": judge_result
            }

        out_path = get_eval_path(inference_path)
        safe_json_dump(eval_result, out_path, indent=2)
        
        return round(final_score, 4), round(acc_score, 4), round(logic_score, 4), decision_score, None
            
    except Exception as e:
        error_msg = f"评估失败: {traceback.format_exc()}"
        logger.error(error_msg)
        return None, None, None, None, error_msg

def main():
    """主函数"""
    logger.info("="*80)
    logger.info("表格预测模型评估系统 (Mode: Split no_tool/with_tool)")
    logger.info("="*80)
    logger.info(f"数据集根目录: {PATH_CONFIG.dataset_root}")
    logger.info(f"推理结果根目录: {PATH_CONFIG.inference_root}")
    logger.info(f"目标Benchmark: {PATH_CONFIG.target_benchmark}")
    logger.info(f"目标模型: {PATH_CONFIG.target_models or '全部'}")
    logger.info("="*80)
    
    # 收集任务
    inference_files = PathManager.collect_inference_files(
        PATH_CONFIG.inference_root,
        PATH_CONFIG.target_models,
        PATH_CONFIG.target_benchmark,
        PATH_CONFIG.target_mode,
        JUDGE_MODEL_ID
    )

    logger.info(f"找到 {len(inference_files)} 个待评估文件")
    if PATH_CONFIG.target_mode:
        logger.info(f"评估模式: {PATH_CONFIG.target_mode}")
    
    if not inference_files:
        logger.warning("未找到任何推理文件")
        return
    
    # 初始化分离的统计容器
    MODES = ["no_tool", "with_tool", "aide_tool_gpt"]
    stats_buckets = {
        mode: {
            "results": [],      # (score, model_name)
            "errors": [],       # (path, model, err)
            "skipped": [],
            "model_results": {} # {model_name: {"final_scores": [], ...}}
        } 
        for mode in MODES
    }

    # 执行评估
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {}
        for inf_path, model_name, mode in inference_files:
            dataset_path = PathManager.get_dataset_path_from_inference(
                inf_path, 
                PATH_CONFIG.inference_root,
                PATH_CONFIG.dataset_root
            )
            future = executor.submit(
                evaluate_single_file, 
                inf_path, 
                dataset_path, 
                model_name, 
                PATH_CONFIG.target_benchmark
            )
            future_to_task[future] = (inf_path, model_name, mode)
        
        with tqdm(total=len(inference_files), desc="评估进度") as pbar:
            for future in as_completed(future_to_task):
                try:
                    inf_path, model_name, mode_val = future_to_task[future]
                    final_score, acc_score, logic_score, decision_score, err = future.result()
                    
                    bucket = stats_buckets[mode_val]
                    if model_name not in bucket["model_results"]:
                        bucket["model_results"][model_name] = {
                            "final_scores": [], "acc_scores": [], "logic_scores": [], 
                            "decision_scores": [], 
                            "errors": [], "skipped": [],
                            "empty_count": 0 # [新增] 空响应计数
                        }
                    m_res = bucket["model_results"][model_name]
                    file_name = os.path.basename(inf_path)

                    if err and err.startswith("SKIPPED"):
                        bucket["skipped"].append((inf_path, model_name, err))
                        m_res["skipped"].append(inf_path)
                        pbar.write(f"⏭️ [{mode_val}|{model_name}] 跳过: {file_name}")
                    
                    # [新增] 处理空响应错误，视为 0 分的有效结果
                    elif err == "EMPTY_RESPONSE":
                        bucket["results"].append((final_score, model_name))
                        m_res["final_scores"].append(final_score)
                        m_res["acc_scores"].append(acc_score)
                        m_res["logic_scores"].append(logic_score)
                        if decision_score is not None:
                            m_res["decision_scores"].append(decision_score)
                        
                        m_res["empty_count"] += 1 # 累加空响应
                        pbar.write(f"⚠️ [{mode_val}|{model_name}] {file_name}: Empty Response -> 0.0")

                    elif err:
                        bucket["errors"].append((inf_path, model_name, err))
                        m_res["errors"].append((inf_path, err))
                        pbar.write(f"❌ [{mode_val}|{model_name}] {file_name}: 错误: {err[:50]}...")
                    
                    elif final_score is not None:
                        bucket["results"].append((final_score, model_name))
                        m_res["final_scores"].append(final_score)
                        m_res["acc_scores"].append(acc_score)
                        m_res["logic_scores"].append(logic_score)
                        
                        if decision_score is not None:
                            m_res["decision_scores"].append(decision_score)
                        
                        dec_str = f"| Dec {decision_score:.0f}" if decision_score is not None else ""
                        pbar.write(f"✓ [{mode_val}|{model_name}] {file_name}: Final {final_score:.2f} {dec_str}")
            
                except Exception as critical_e:
                    logger.error(f"Critical Loop Error: {critical_e}")
                finally:
                    pbar.update(1)

    # 3. 统计、保存报告和打印
    def calc_stats(scores: List[float]) -> Dict:
        if not scores: return {"avg": 0.0, "max": 0.0, "min": 0.0}
        return {
            "avg": round(sum(scores) / len(scores), 4),
            "max": round(max(scores), 4),
            "min": round(min(scores), 4)
        }
    
    table_rows = []
    
    for mode in MODES:
        bucket = stats_buckets[mode]
        model_results = bucket["model_results"]
        skipped_global = bucket["skipped"]
        
        for model_name, m_data in model_results.items():
            f_scores = m_data["final_scores"]
            errors = m_data["errors"]
            model_skipped = m_data.get("skipped", [])
            empty_count = m_data.get("empty_count", 0) # [新增]
            
            if not f_scores and not errors and not model_skipped: 
                continue

            # 3.1 计算统计量
            stat_final = calc_stats(f_scores)
            stat_acc = calc_stats(m_data["acc_scores"])
            stat_logic = calc_stats(m_data["logic_scores"])
            stat_dec = calc_stats(m_data["decision_scores"]) 
            has_dec = len(m_data["decision_scores"]) > 0

            # 3.2 确定保存目录
            target_report_dir = os.path.join(PATH_CONFIG.inference_root, model_name, mode)
            os.makedirs(target_report_dir, exist_ok=True)

            # 3.3 构建 JSON 汇总报告
            model_report = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": model_name,
                "mode": mode,
                "benchmark": PATH_CONFIG.target_benchmark,
                "counts": {
                    "total": len(f_scores) + len(errors) + len(model_skipped),
                    "success": len(f_scores) - empty_count, # 纯成功
                    "empty": empty_count, # [新增] 空响应
                    "failed": len(errors),
                    "skipped": len(model_skipped)
                },
                "stats": {
                    "final_score": stat_final,
                    "accuracy_score": stat_acc, 
                    "logic_score": stat_logic,
                    "decision_score": stat_dec if has_dec else None 
                }
            }

            # 写入 JSON
            json_filename = f"summary_{PATH_CONFIG.target_benchmark}_{model_name}_{mode}.json"
            report_path = os.path.join(target_report_dir, json_filename)
            safe_json_dump(model_report, report_path)
            logger.info(f"已保存报告: {report_path}")

            # 3.4 保存 CSV 错误日志
            if errors:
                try:
                    error_df = pd.DataFrame(errors, columns=["file_path", "error_message"])
                    csv_filename = f"errors_{PATH_CONFIG.target_benchmark}_{model_name}_{mode}.csv"
                    error_df.to_csv(os.path.join(target_report_dir, csv_filename), index=False)
                except Exception as e:
                    logger.error(f"保存错误日志失败: {e}")

            # 3.5 收集打印用的表格数据
            table_rows.append({
                "Model": model_name, "Mode": mode,
                "Total": len(f_scores) + len(errors) + len(model_skipped),
                "Success": len(f_scores), # 包含 empty (因为是0分有效值)
                "Empty": empty_count, # [新增]
                "Errors": len(errors),
                "Skipped": len(model_skipped),
                "Avg Final": stat_final["avg"],
                "Avg Acc": stat_acc["avg"],
                "Avg Logic": stat_logic["avg"],
                "Avg Dec": stat_dec["avg"] if has_dec else None
            })

        # 3.6 保存该模式下的全局跳过日志
        if skipped_global:
            try:
                skip_df = pd.DataFrame(skipped_global, columns=["file_path", "model", "reason"])
                csv_filename = f"skipped_{PATH_CONFIG.target_benchmark}_{mode}.csv"
                skip_df.to_csv(os.path.join(PATH_CONFIG.inference_root, csv_filename), index=False)
            except Exception as e:
                logger.error(f"保存跳过日志失败: {e}")

    # ================= 打印报表 =================
    if table_rows:
        print("\n" + "="*170)
        print(f"{'FINAL EVALUATION REPORT':^170}")
        print("="*170)
        
        # [修改] 增加 Empty 列
        header_fmt = "| {:<25} | {:<10} | {:<6} | {:<6} | {:<6} | {:<6} | {:<6} | {:<10} | {:<10} | {:<10} | {:<10} |"
        print(header_fmt.format("Model", "Mode", "Total", "Succ", "Empty", "Err", "Skip", "Avg Final", "Avg Acc", "Avg Logic", "Avg Dec"))
        print("-" * 170)
        
        for row in table_rows:
            dec_str = f"{row['Avg Dec']:.4f}" if row['Avg Dec'] is not None else "-"
            print(header_fmt.format(
                row["Model"][:25], row["Mode"],
                str(row["Total"]), str(row["Success"]), str(row["Empty"]), str(row['Errors']), str(row['Skipped']),
                f"{row['Avg Final']:.4f}", f"{row['Avg Acc']:.4f}", f"{row['Avg Logic']:.4f}", 
                dec_str
            ))
        print("-" * 170)
        print("* Succ includes Empty (0.0 score). Total = Succ + Err + Skip.")
        print("* Avg Acc: B1(Pred), B2(Avg Attr), B3(Pred 002)")
        print("* Avg Dec: B2(Decision Hit), B3(Trend Hit)")
        print("="*170 + "\n")
    else:
        logger.warning("没有生成任何有效结果，无法打印报告。")

    if DUMP_JUDGE_ARTIFACTS:
        export_judge_artifacts()

    analysis_records = []
    analysis_inference_files = inference_files
    if ANALYSIS_USE_ALL_INFERENCE_FILES and (
        EXPORT_SCALE_BREAKDOWN or EXPORT_SHAPE_BREAKDOWN or EXPORT_METRIC_SENSITIVITY or EXPORT_METRIC_SENSITIVITY_FULL
    ):
        analysis_inference_files = PathManager.collect_inference_files(
            PATH_CONFIG.inference_root,
            PATH_CONFIG.target_models,
            PATH_CONFIG.target_benchmark,
            PATH_CONFIG.target_mode,
            JUDGE_MODEL_ID,
            skip_existing_eval=False,
        )

    if EXPORT_SCALE_BREAKDOWN or EXPORT_SHAPE_BREAKDOWN or EXPORT_METRIC_SENSITIVITY or EXPORT_METRIC_SENSITIVITY_FULL:
        analysis_records = load_per_file_analysis_records(analysis_inference_files)

    if EXPORT_SCALE_BREAKDOWN:
        export_scale_breakdown(analysis_records)

    if EXPORT_SHAPE_BREAKDOWN:
        export_shape_breakdown(analysis_records)

    if EXPORT_METRIC_SENSITIVITY:
        export_metric_sensitivity(analysis_inference_files)

    if EXPORT_METRIC_SENSITIVITY_FULL:
        export_metric_sensitivity(analysis_inference_files, full_grid=True)

    logger.info("评估流程结束。")

if __name__ == "__main__":
    main()

# python eval_b2_v1.py --benchmark B1 --models deepseek --mode aide_tool_gpt
