import os
import json
import pandas as pd
import numpy as np
import tiktoken
import subprocess
import uuid
import threading
import time
import random
import shutil
import gc
import argparse
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import tempfile
from pathlib import Path

# ================= 模型配置部分 =================


def _env_keys(name: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]


LEGACY_DATA_ROOT = Path(os.getenv("TOPBENCH_LEGACY_DATA_ROOT", Path.cwd())).resolve()
SANDBOX_IMAGE = os.getenv("TOPBENCH_SANDBOX_IMAGE", "topbench-sandbox:latest")


# GPT 配置
GPT_API_KEYS = _env_keys("OPENAI_API_KEY")
GPT_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
GPT_MODEL_ID = os.getenv("OPENAI_MODEL_ID", "gpt-5.2")

# DeepSeek 配置
DS_API_KEYS = _env_keys("DEEPSEEK_API_KEY")
DS_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DS_MODEL_ID = os.getenv("DEEPSEEK_MODEL_ID", "deepseek-chat")
DS_REASON_MODEL_ID = os.getenv("DEEPSEEK_REASONER_MODEL_ID", "deepseek-reasoner")

# Qwen 配置
QWEN_API_KEYS = _env_keys("QWEN_API_KEY")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL_ID = os.getenv("QWEN_MODEL_ID", "qwen3-235b-a22b-instruct-2507")
QWEN_THINK_MODEL_ID = os.getenv("QWEN_THINK_MODEL_ID", "qwen3-235b-a22b-thinking-2507")

# Claude 配置 (通过 OpenRouter)
CLAUDE_API_KEYS = _env_keys("ANTHROPIC_API_KEY") or _env_keys("OPENROUTER_API_KEY")
CLAUDE_BASE_URL = os.getenv("CLAUDE_BASE_URL", "https://openrouter.ai/api/v1")
CLAUDE_MODEL_ID = os.getenv("CLAUDE_MODEL_ID", "anthropic/claude-sonnet-4.5")
CLAUDE_SITE_URL = os.getenv("CLAUDE_SITE_URL", "https://localhost")
CLAUDE_SITE_NAME = os.getenv("CLAUDE_SITE_NAME", "TablePredictBench")

# Gemini 配置
GEMINI_API_KEYS = _env_keys("GEMINI_API_KEY")
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-3-flash-preview")

# 全局变量（在运行时根据选择的模型设置）
API_KEYS = []
BASE_URL = ""
MODEL_ID = ""
key_iterator = None
CURRENT_MODEL_TYPE = ""  # 'openai', 'gemini'


def is_query_json_for_inference(filename: str, *, requires_current: bool = False) -> bool:
    if not filename.endswith(".json"):
        return False
    excluded_names = {"info.json", "info_mod.json", "stats_cache.json", "extracted_features.json"}
    if filename in excluded_names:
        return False
    excluded_markers = ("_eval", "_no_tool", "_with_tool", "_text_reasoning", "_agentic_workflow")
    if any(marker in filename for marker in excluded_markers):
        return False
    if requires_current and "_current" not in filename:
        return False
    return True

# 任务配置映射
TASK_CONFIGS = {
    "B1": {
        "base_dir": str(LEGACY_DATA_ROOT / "B1andB3" / "B1"),
        "categories": ["daily", "finance", "medical"],
        "mode": "no_tool",
        "has_current": False,
        "context_size": 48000,
        "file_filter": lambda f: is_query_json_for_inference(f)
    },
    "B2": {
        "base_dir": str(LEGACY_DATA_ROOT / "B2"),
        "categories": ["daily", "finance", "medical"],
        "mode": "no_tool",
        "has_current": False,
        "context_size": 48000,
        "file_filter": lambda f: is_query_json_for_inference(f)
    },
    "B3": {
        "base_dir": str(LEGACY_DATA_ROOT / "B3"),
        "categories": ["daily", "finance", "medical"],
        "mode": "no_tool",
        "has_current": False,
        "context_size": 48000,
        "file_filter": lambda f: is_query_json_for_inference(f)
    },
    "B4": {
        "base_dir": str(LEGACY_DATA_ROOT / "B4"),
        "categories": ["daily", "finance", "medical"],
        "mode": "no_tool",
        "has_current": True,
        "context_size": 48000,
        "file_filter": lambda f: is_query_json_for_inference(f, requires_current=True)
    }
}

# 全局配置
MAX_WORKERS = 1
RESERVED_TOKENS = 0
MAX_ITERATIONS = 10
TRUNCATION_STRATEGY = "random"
TRUNCATION_RANDOM_SEED = 42
# google-genai HttpOptions.timeout uses milliseconds.
GEMINI_REQUEST_TIMEOUT_MS = 900000
GEMINI_DISCONNECT_FALLBACK_RATIOS = (0.8, 0.5)

# ================= 工具定义 =================

# OpenAI 格式工具定义
tools_openai = [
    {
        "type": "function",
        "function": {
            "name": "CodeRunner",
            "description": "Executes Python code. CRITICAL: output valid JSON. Escape quotes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {"type": "string", "enum": ["python"]},
                    "code": {"type": "string", "description": "Python code to execute. CRITICAL: The code executed in each call is independent; do not omit any code based on the previous call. The whole code must be provided each time."}
                },
                "required": ["language", "code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "PipInstaller",
            "description": "Installs Python packages via pip.",
            "parameters": {
                "type": "object",
                "properties": {
                    "package_name": {"type": "string", "description": "Package name (e.g. 'scikit-learn')."}
                },
                "required": ["package_name"]
            }
        }
    }
]

# Gemini 格式工具定义
try:
    from google import genai
    from google.genai import types
    
    tools_gemini = [
        types.FunctionDeclaration(
            name="CodeRunner",
            description="Executes Python code in a sandboxed environment.",
            parameters={
                "type": "object",
                "properties": {
                    "language": {"type": "string", "enum": ["python"]},
                    "code": {"type": "string", "description": "Python code to execute."}
                },
                "required": ["language", "code"]
            }
        ),
        types.FunctionDeclaration(
            name="PipInstaller",
            description="Installs Python packages via pip.",
            parameters={
                "type": "object",
                "properties": {
                    "package_name": {"type": "string", "description": "Package name."}
                },
                "required": ["package_name"]
            }
        )
    ]
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-genai not installed. Gemini model will not be available.")

# ================= 模型初始化函数 =================

def initialize_model(model_name):
    """根据模型名称初始化全局配置"""
    global API_KEYS, BASE_URL, MODEL_ID, key_iterator, CURRENT_MODEL_TYPE
    
    model_map = {
        "gpt": (GPT_API_KEYS, GPT_BASE_URL, GPT_MODEL_ID, "openai"),
        "chatgpt52": (GPT_API_KEYS, GPT_BASE_URL, GPT_MODEL_ID, "openai"),
        "deepseek": (DS_API_KEYS, DS_BASE_URL, DS_MODEL_ID, "openai"),
        "qwen": (QWEN_API_KEYS, QWEN_BASE_URL, QWEN_MODEL_ID, "openai"),
        "deepseek_reasoner": (DS_API_KEYS, DS_BASE_URL, DS_REASON_MODEL_ID, "openai"),
        "qwen_thinking": (QWEN_API_KEYS, QWEN_BASE_URL, QWEN_THINK_MODEL_ID, "openai"),
        "claude": (CLAUDE_API_KEYS, CLAUDE_BASE_URL, CLAUDE_MODEL_ID, "openai"),
        "gemini": (GEMINI_API_KEYS, None, GEMINI_MODEL_ID, "gemini"),
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(model_map.keys())}")
    
    if model_name == "gemini" and not GEMINI_AVAILABLE:
        raise ImportError("google-genai package is required for Gemini. Install: pip install google-genai")
    
    API_KEYS, BASE_URL, MODEL_ID, CURRENT_MODEL_TYPE = model_map[model_name]
    if not API_KEYS:
        raise ValueError(f"No API key configured for model '{model_name}'. Check the corresponding environment variable.")
    key_iterator = itertools.cycle(API_KEYS)
    
    print(f">>> Initialized model: {model_name}")
    print(f"    Model ID: {MODEL_ID}")
    print(f"    Base URL: {BASE_URL}")
    print(f"    API Keys: {len(API_KEYS)} keys loaded")

# ================= 沙盒执行函数 =================

def execute_python_code(code: str, mount_files: dict = None, packages: list = None, artifact_dest_path: str = None):
    """执行 Python 代码，支持从沙盒中提取 result.csv"""
    task_id = str(uuid.uuid4())[:8]
    
    preamble_code = ""
    if packages and len(packages) > 0:
        pkg_str = " ".join(packages)
        preamble_code = (
            f"import subprocess, sys\n"
            f"try:\n"
            f"    subprocess.check_call(\n"
            f"        [sys.executable, '-m', 'pip', 'install', '{pkg_str}', '-i', 'https://pypi.tuna.tsinghua.edu.cn/simple'],\n"
            f"        stdout=subprocess.DEVNULL,\n"
            f"        stderr=subprocess.DEVNULL\n"
            f"    )\n"
            f"    print('>>> Install complete.')\n"
            f"except Exception as e:\n"
            f"    print(f'>>> Install failed: {{e}}')\n"
            f"\n"
        )
    
    full_code = preamble_code + code

    sandbox_tmp_parent = os.getenv("TOPBENCH_SANDBOX_TMPDIR", "").strip()
    if sandbox_tmp_parent:
        os.makedirs(sandbox_tmp_parent, exist_ok=True)
        temp_context = tempfile.TemporaryDirectory(dir=sandbox_tmp_parent)
    else:
        temp_context = tempfile.TemporaryDirectory()

    with temp_context as temp_dir:
        script_name = f"script_{task_id}.py"
        host_script_path = os.path.join(temp_dir, script_name)
        
        with open(host_script_path, "w", encoding='utf-8') as f:
            f.write(full_code)
        
        if mount_files:
            for dest_name, src_path in mount_files.items():
                if os.path.exists(src_path):
                    try:
                        shutil.copy(src_path, os.path.join(temp_dir, dest_name))
                    except Exception as e:
                        return f"System Error (Copy {dest_name} failed): {str(e)}"

        cmd = [
            "docker", "run", "--rm",
            "--memory=8g",
            "-v", f"{temp_dir}:/app",
            "-w", "/app",
            SANDBOX_IMAGE,
            "python", script_name
        ]
        
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                encoding='utf-8'
            )
            # === 新增逻辑：提取 result.csv ===
            if artifact_dest_path:
                docker_csv_path = os.path.join(temp_dir, "result.csv")
                if os.path.exists(docker_csv_path):
                    try:
                        shutil.copy(docker_csv_path, artifact_dest_path)
                        # print(f"Successfully extracted result.csv to {artifact_dest_path}")
                    except Exception as copy_err:
                        print(f"Failed to copy artifact: {copy_err}")
            # ==============================
            
            if result.returncode == 0:
                return result.stdout.strip() or "Code executed with no output.If you want to get some info, please use 'print()'."
            else:
                return f"Execution Error:\n{result.stderr}\n\nStandard Output:\n{result.stdout[:10000]}"
                
        except subprocess.TimeoutExpired:
            return "Error: Execution timed out."
        except Exception as e:
            return f"System Error: {str(e)}"

# ================= 日志函数 =================

_log_lock = threading.Lock()

def append_to_log(log_path: str, content: str) -> None:
    with _log_lock:
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(content + "\n")
        except Exception as e:
            print(f"[Log Error] {e}")

def save_llm_step_to_log(log_path, step_idx, message, duration=0.0):
    output = []
    output.append(f"\n{'='*20} Step {step_idx} Response (Time: {duration:.2f}s) {'='*20}")
    
    # OpenAI 格式
    if hasattr(message, 'content'):
        if message.content:
            output.append(f"\n[Content]:\n{message.content}")
        if hasattr(message, 'tool_calls') and message.tool_calls:
            output.append(f"\n[Tool Calls]:")
            for tc in message.tool_calls:
                output.append(f"  - Function: {tc.function.name}")
                output.append(f"  - Arguments: {tc.function.arguments}")
    
    # Gemini 格式
    elif hasattr(message, 'text'):
        output.append(f"\n[Content]:\n{message.text}")
        if hasattr(message, 'candidates'):
            for candidate in message.candidates:
                if candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            output.append(f"\n[Tool Call]: {part.function_call.name}")
    
    output.append(f"{'='*55}\n")
    append_to_log(log_path, "".join(output))

# ================= API 调用函数 =================

def generate_completion_openai(log_path: str, max_retries: int = 10, base_delay: float = 5.0, **kwargs):
    """OpenAI 兼容 API 调用"""
    from openai import OpenAI, APIStatusError, RateLimitError, AuthenticationError, APIConnectionError, Timeout
    
    # if "max_tokens" not in kwargs:
    #     kwargs["max_tokens"] = 8192

    for attempt in range(max_retries):
        current_api_key = next(key_iterator)
        masked_key = current_api_key[:6] + "..." + current_api_key[-4:]
        
        extra_headers = {}
        if CURRENT_MODEL_TYPE == "openai" and "claude" in MODEL_ID.lower():
            extra_headers = {
                "HTTP-Referer": CLAUDE_SITE_URL,
                "X-Title": CLAUDE_SITE_NAME,
            }
        
        client = OpenAI(
            api_key=current_api_key,
            base_url=BASE_URL,
            timeout=Timeout(connect=10.0, read=1800.0, write=1800.0, pool=1800.0),
        )
        
        try:
            if extra_headers:
                return client.chat.completions.create(extra_headers=extra_headers, **kwargs)
            else:
                return client.chat.completions.create(**kwargs)
        except AuthenticationError:
            append_to_log(log_path, f"[Auth Failed] Key {masked_key} is invalid")
            time.sleep(1)
        except (RateLimitError, APIStatusError, APIConnectionError) as e:
            delay = base_delay * (1.5 ** attempt) + random.uniform(1, 3)
            resp_content = "N/A"
            if hasattr(e, 'response') and e.response is not None:
                # 尝试提取响应文本
                resp_content = str(getattr(e.response, 'text', str(e.response)))
            # 添加完整错误信息
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "status_code": getattr(e, 'status_code', None),
                "response": resp_content
            }
            append_to_log(log_path, f"[FULL ERROR] {json.dumps(error_details, indent=2)}")
            time.sleep(delay)
    
    raise Exception(f"Max retries ({max_retries}) exceeded")

def generate_completion_gemini(log_path: str, contents: list, system_instruction: str, 
                               tools_config: list = None, max_retries: int = 10, base_delay: float = 5.0):
    """Gemini API 调用"""
    current_system_instruction = system_instruction
    disconnect_count = 0
    fallback_stage = 0

    for attempt in range(max_retries):
        current_api_key = next(key_iterator)
        http_options = types.HttpOptions(timeout=GEMINI_REQUEST_TIMEOUT_MS)
        client = genai.Client(api_key=current_api_key, http_options=http_options)
        
        gemini_tools = []
        if tools_config:
            gemini_tools = [types.Tool(function_declarations=tools_config)]
        
        config = types.GenerateContentConfig(
            httpOptions=http_options,
            tools=gemini_tools,
            system_instruction=current_system_instruction,
            # thinking_config=types.ThinkingConfig(thinking_level="low"),
            temperature=0.1
        )
        
        try:
            return client.models.generate_content(
                model=MODEL_ID,
                contents=contents,
                config=config
            )
        except Exception as e:
            error_text = str(e)
            if "Server disconnected without sending a response" in error_text:
                disconnect_count += 1
                if disconnect_count >= 2 and fallback_stage < len(GEMINI_DISCONNECT_FALLBACK_RATIOS):
                    target_ratio = GEMINI_DISCONNECT_FALLBACK_RATIOS[fallback_stage]
                    reduced_instruction = _reduce_history_prompt_context(
                        current_system_instruction,
                        ratio=target_ratio,
                    )
                    if reduced_instruction != current_system_instruction:
                        current_system_instruction = reduced_instruction
                        fallback_stage += 1
                        append_to_log(
                            log_path,
                            f"[Gemini Fallback] After {disconnect_count} disconnects, reduced serialized table context to {int(target_ratio * 100)}% and retrying.",
                        )
                        disconnect_count = 0
                    elif fallback_stage < len(GEMINI_DISCONNECT_FALLBACK_RATIOS):
                        fallback_stage += 1
            delay = base_delay * (1.5 ** attempt) + random.uniform(1, 3)
            append_to_log(log_path, f"[Gemini Error] Retry in {delay:.1f}s | ERROR: {e}")
            time.sleep(delay)
    
    raise Exception(f"Max retries ({max_retries}) exceeded")

# ================= 核心推理函数 =================

def run_inference_single_csv(query, table_content, file_abs_path, mode, log_path):
    """单个 CSV 文件的推理逻辑"""
    append_to_log(log_path, f">>> Mode: {mode} | Query: {query}")

    metrics = {
        "step_details": [],
        "code_total_count": 0,
        "code_success_count": 0,
        "total_steps": 0
    }

    if mode == "no_tool":
        system_content = f"Here is the preview/content of the history data:\n\n{table_content}"
        user_content = query
    else:
        system_content = (
            f"The history data file is located at: history.csv\n"
            f"(Note: The file is mounted in your environment, use 'history.csv' directly)\n"
            f"Here are the columns of the file:\n[{table_content}]\n\n"
            f"IMPORTANT: You MUST use the 'CodeRunner' tool to read the file to inspect the data content.\n"
            f"You need to give the answer within {MAX_ITERATIONS} rounds."
        )
        user_content = query

    append_to_log(log_path, f"\n[{'='*20} System Prompt {'='*20}]\n{system_content}\n{'='*55}\n")

    installed_packages = []

    if mode == "no_tool":
        try:
            t_start = time.time()
            
            if CURRENT_MODEL_TYPE == "gemini":
                response = generate_completion_gemini(
                    log_path=log_path,
                    contents=[user_content],
                    system_instruction=system_content
                )
                content = response.text
            else:
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ]
                response = generate_completion_openai(
                    log_path=log_path, model=MODEL_ID, messages=messages, temperature=0.1
                )
                content = response.choices[0].message.content
            
            t_end = time.time()
            duration = t_end - t_start
            
            save_llm_step_to_log(log_path, 1, response if CURRENT_MODEL_TYPE == "gemini" else response.choices[0].message, duration)
            metrics["step_details"].append({"step": 1, "type": "llm", "duration_seconds": duration})
            metrics["total_steps"] = 1
            
            return content, metrics
        except Exception as e:
            err = f"Error in no_tool mode: {e}"
            append_to_log(log_path, err)
            return err, metrics
    else:
        mount_files = {"history.csv": file_abs_path} # 显式定义挂载
        if CURRENT_MODEL_TYPE == "gemini":
            return run_inference_gemini_tool(query, system_content, mount_files, log_path, metrics)
        else:
            return run_inference_openai_tool(query, system_content, mount_files, log_path, metrics)

def run_inference_openai_tool(query, system_content, mount_files, log_path, metrics, csv_output_path=None):
    """OpenAI 格式的 Tool 模式推理"""
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query}
    ]
    installed_packages = []

    for i in range(MAX_ITERATIONS):
        try:
            t_llm_start = time.time()
            response = generate_completion_openai(
                log_path=log_path, model=MODEL_ID, messages=messages,
                tools=tools_openai, tool_choice="auto", temperature=0.1
            )
            t_llm_end = time.time()
            llm_duration = t_llm_end - t_llm_start
            
            msg = response.choices[0].message
            save_llm_step_to_log(log_path, i + 1, msg, llm_duration)
            metrics["step_details"].append({"step": i+1, "type": "llm", "duration_seconds": llm_duration})
            
            if not msg.tool_calls:
                metrics["total_steps"] = i + 1
                return msg.content, metrics
            
            messages.append(msg)
            
            for tool in msg.tool_calls:
                tool_result_content = ""
                t_tool_start = time.time()
                try:
                    # 先判断工具类型，如果是 CodeRunner，无论后续解析是否成功，先计入 total_count
                    is_code_runner = (tool.function.name == "CodeRunner")
                    if is_code_runner:
                        metrics["code_total_count"] += 1

                    # 尝试解析参数
                    try:
                        args = json.loads(tool.function.arguments)
                    except Exception:
                        # 针对特定的格式错误抛出明确异常，以便记录到 System Error
                        raise ValueError(f"Invalid JSON arguments: {tool.function.arguments}")
                    if tool.function.name == "PipInstaller":
                        pkg_name = args.get("package_name")
                        append_to_log(log_path, f"      [Pip] Queued: {pkg_name}")
                        if pkg_name:
                            installed_packages.append(pkg_name)
                            installed_packages = list(set(installed_packages))
                            tool_result_content = f"Package '{pkg_name}' added."
                        else:
                            tool_result_content = "Error: package_name missing."

                    elif tool.function.name == "CodeRunner":
                        code = args.get("code")
                        if not code:
                            raise ValueError("Code missing.")
                        
                        # mount_files = {"history.csv": file_abs_path}
                        append_to_log(log_path, f"      [Sandbox] Executing...\n{code}")
                        # === 修改调用 ===
                        execution_output = execute_python_code(
                            code, 
                            mount_files=mount_files, 
                            packages=installed_packages,
                            artifact_dest_path=csv_output_path # 传入保存路径
                        )
                        # ===============
                        append_to_log(log_path, f"      [Sandbox Output]:\n{execution_output}")
                        tool_result_content = str(execution_output[:100000])
                        
                        if "Execution Error:" not in tool_result_content and "System Error" not in tool_result_content:
                            metrics["code_success_count"] += 1

                except Exception as e:
                    tool_result_content = f"System Error: {str(e)}"

                t_tool_end = time.time()
                tool_duration = t_tool_end - t_tool_start
                metrics["step_details"].append({"step": i+1, "type": "tool_execution", "duration_seconds": tool_duration})

                messages.append({
                    "tool_call_id": tool.id,
                    "role": "tool",
                    "name": tool.function.name,
                    "content": tool_result_content
                })
        
        except Exception as outer_e:
            err_str = f"Critical API Error: {outer_e}"
            append_to_log(log_path, err_str)
            metrics["total_steps"] = i + 1
            return err_str, metrics
    
    try:
        append_to_log(log_path, "\n[INFO] Max iterations reached, attempting final response...")
        
        # ==================== 修改开始 ====================
        # 添加一条强制结束的提示，要求模型基于当前信息进行总结
        final_instruction = (
            "You have reached the maximum number of steps allowed. "
            "Do NOT use any more tools (CodeRunner/PipInstaller). "
            "Based on the information you have gathered so far, please summarize your findings "
            "and provide a final answer to the user's original query."
        )
        
        # 将这条指令追加到对话历史中
        # 注意：这里使用 messages + [...] 创建新列表，或者直接 messages.append 都可以
        # 为了不影响外部引用，这里建议创建一个用于本次生成的临时列表
        final_messages = messages + [{"role": "user", "content": final_instruction}]
        # ==================== 修改结束 ====================

        t_final_start = time.time()
        response = generate_completion_openai(
            log_path=log_path, 
            model=MODEL_ID, 
            messages=final_messages,  # <--- 使用带有提示的新消息列表
            temperature=0.1
        )
        t_final_end = time.time()
        final_duration = t_final_end - t_final_start
        
        final_msg = response.choices[0].message
        save_llm_step_to_log(log_path, MAX_ITERATIONS + 1, final_msg, final_duration)
        metrics["step_details"].append({"step": MAX_ITERATIONS + 1, "type": "llm_final", "duration_seconds": final_duration})
        metrics["total_steps"] = MAX_ITERATIONS + 1
        
        return final_msg.content or "Max iterations reached, but got final response.", metrics
    except Exception as e:
        append_to_log(log_path, f"[WARN] Final response failed: {e}")
        metrics["total_steps"] = MAX_ITERATIONS
        return "Max iterations reached.", metrics

def run_inference_gemini_tool(query, system_content, mount_files, log_path, metrics, csv_output_path=None):
    """Gemini 格式的 Tool 模式推理"""
    contents = [query]
    installed_packages = []

    for i in range(MAX_ITERATIONS):
        try:
            t_llm_start = time.time()
            response = generate_completion_gemini(
                log_path=log_path,
                contents=contents,
                system_instruction=system_content,
                tools_config=tools_gemini
            )
            t_llm_end = time.time()
            llm_duration = t_llm_end - t_llm_start
            
            save_llm_step_to_log(log_path, i + 1, response, llm_duration)
            metrics["step_details"].append({"step": i+1, "type": "llm", "duration_seconds": llm_duration})
            
            function_calls = []
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_calls.append(part.function_call)
            
            if not function_calls:
                metrics["total_steps"] = i + 1
                return response.text, metrics
            
            model_turn = response.candidates[0].content
            contents.append(model_turn)
            
            for call in function_calls:
                tool_result_content = ""
                t_tool_start = time.time()
                
                try:
                    args = call.args
                    func_name = call.name
                    is_code_runner = (func_name == "CodeRunner")

                    if is_code_runner:
                        metrics["code_total_count"] += 1
                    
                    if func_name == "PipInstaller":
                        pkg_name = args.get("package_name")
                        append_to_log(log_path, f"      [Pip] Queued: {pkg_name}")
                        if pkg_name:
                            installed_packages.append(pkg_name)
                            installed_packages = list(set(installed_packages))
                            tool_result_content = f"Package '{pkg_name}' added."
                        else:
                            tool_result_content = "Error: package_name missing."

                    elif func_name == "CodeRunner":
                        code = args.get("code")
                        if not code:
                            raise ValueError("Code missing.")

                        metrics["code_total_count"] += 1
                        # mount_files = {"history.csv": file_abs_path}
                        append_to_log(log_path, f"      [Sandbox] Executing...\n{code}")
                        # === 修改调用 ===
                        execution_output = execute_python_code(
                            code, 
                            mount_files=mount_files, 
                            packages=installed_packages,
                            artifact_dest_path=csv_output_path # 传入保存路径
                        )
                        # ===============
                        append_to_log(log_path, f"      [Sandbox Output]:\n{execution_output}")
                        tool_result_content = str(execution_output[:100000])
                        
                        if "Execution Error" not in tool_result_content and "System Error" not in tool_result_content:
                            metrics["code_success_count"] += 1

                except Exception as e:
                    tool_result_content = f"System Error: {str(e)}"

                t_tool_end = time.time()
                tool_duration = t_tool_end - t_tool_start
                metrics["step_details"].append({"step": i+1, "type": "tool_execution", "duration_seconds": tool_duration})

                response_part = types.Part.from_function_response(
                    name=func_name,
                    response={"result": tool_result_content}
                )
                contents.append(types.Content(role="user", parts=[response_part]))
        
        except Exception as outer_e:
            err_str = f"Critical API Error: {outer_e}"
            append_to_log(log_path, err_str)
            metrics["total_steps"] = i + 1
            return err_str, metrics
    
    try:
        append_to_log(log_path, "\n[INFO] Max iterations reached, attempting final response...")
        
        # ==================== 修改开始 ====================
        # 1. 构造强制总结的提示词
        final_instruction = (
            "You have reached the maximum number of steps allowed. "
            "Do NOT use any more tools. "
            "Based on the information you have gathered so far, please summarize your findings "
            "and provide a final answer to the user's original query."
        )
        
        # 2. 封装为 Gemini 的 Content 对象 (假设上下文中有 types 引用)
        final_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=final_instruction)]
        )
        
        # 3. 创建包含该指令的新历史列表
        final_contents = contents + [final_content]
        # ==================== 修改结束 ====================

        t_final_start = time.time()
        
        # 4. 使用 final_contents 发起最后一次调用
        # 注意：这里不传 tools_config，确保模型无法再调用工具
        response = generate_completion_gemini(
            log_path=log_path,
            contents=final_contents,  # <--- 使用新的列表
            system_instruction=system_content
        )
        t_final_end = time.time()
        final_duration = t_final_end - t_final_start
        
        save_llm_step_to_log(log_path, MAX_ITERATIONS + 1, response, final_duration)
        metrics["step_details"].append({"step": MAX_ITERATIONS + 1, "type": "llm_final", "duration_seconds": final_duration})
        metrics["total_steps"] = MAX_ITERATIONS + 1
        
        return response.text or "Max iterations reached, but got final response.", metrics
    except Exception as e:
        append_to_log(log_path, f"[WARN] Final response failed: {e}")
        metrics["total_steps"] = MAX_ITERATIONS
        return "Max iterations reached.", metrics

def run_inference_dual_csv(query, history_text, current_text, mount_files, mode, log_path, inner_task_type=None, csv_output_path=None):
    """双 CSV 文件的推理逻辑（B4）"""
    append_to_log(log_path, f">>> Mode: {mode} | Query: {query}")
    
    metrics = {
        "step_details": [],
        "code_total_count": 0,
        "code_success_count": 0,
        "total_steps": 0
    }

    if mode == "no_tool":
        system_content = f"Here is the preview/content of the history data:\n\n{history_text}"
        user_content = f"{query}\n\n{current_text}"
    else:
        # === 修改 System Prompt ===
        prompt_extras = ""
        if inner_task_type and str(inner_task_type).lower() == "regression":
            prompt_extras = "\nWARNING: This is a REGRESSION task. When outputting the list, strictly verify the Top-K order based on your estimated values (Descending/Ascending as required)."

        system_content = (
            f"You have access to two csv files in your environment:\n"
            f"1. 'history.csv'\n"
            f"   - Columns: {history_text}\n"
            f"2. 'current.csv'\n"
            f"   - Columns: {current_text}\n\n"
            f"Note: These files are mounted, use their filenames directly.\n"
            f"The data provided above are ONLY column names. DO NOT hallucinate data rows.\n"
            f"You MUST use the CodeRunner tool to read the files (e.g., pd.read_csv) to inspect the actual data content.\n\n"
            f"CRITICAL REQUIREMENT:\n"
            f"1. You MUST process the data and save the final results into a file named 'result.csv'.\n"
            f"2. The 'result.csv' MUST contain the exact columns matching history.csv format.\n"
            f"3. Do not just print the result, you must save it to 'result.csv' using pandas to_csv()."
            f"{prompt_extras}\n"
            f"You need to give the answer within {MAX_ITERATIONS} rounds."
        )
        user_content = f"{query}"
    
    append_to_log(log_path, f"\n[{'='*20} System Prompt {'='*20}]\n{system_content}\n")

    if mode == "no_tool":
        try:
            t_start = time.time()
            if CURRENT_MODEL_TYPE == "gemini":
                response = generate_completion_gemini(
                    log_path=log_path,
                    contents=[user_content],
                    system_instruction=system_content
                )
                content = response.text
                msg_obj = response
            else:
                messages = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]
                response = generate_completion_openai(log_path=log_path, model=MODEL_ID, messages=messages, temperature=0.1)
                content = response.choices[0].message.content
                msg_obj = response.choices[0].message

            t_end = time.time()
            duration = t_end - t_start
            save_llm_step_to_log(log_path, 1, msg_obj, duration)
            metrics["step_details"].append({"step": 1, "type": "llm", "duration_seconds": duration})
            metrics["total_steps"] = 1
            return content, metrics
        except Exception as e:
            err = f"Error in no_tool mode: {e}"
            append_to_log(log_path, err)
            return err, metrics
    else:
        if CURRENT_MODEL_TYPE == "gemini":
            return run_inference_gemini_tool(user_content, system_content, mount_files, log_path, metrics, csv_output_path=csv_output_path)
        else:
            return run_inference_openai_tool(user_content, system_content, mount_files, log_path, metrics, csv_output_path=csv_output_path)
        
# ================= 文件读取 =================

def get_file_columns(filepath: str) -> str:
    """读取CSV列名"""
    try:
        df = pd.read_csv(filepath, nrows=0, encoding='utf-8', encoding_errors='replace', on_bad_lines='skip', dtype=str, keep_default_na=False)
        return ", ".join([str(c) for c in df.columns.tolist()])
    except Exception as e:
        return f"Error: {str(e)}"

def _fallback_read_head(filepath, max_tokens):
    """备用读取方法：只读取前 1000 行"""
    try:
        df = pd.read_csv(filepath, nrows=1000, encoding_errors='replace', on_bad_lines='skip')
        content = _serialize_csv_with_header(df)
        tokens = TOKENIZER.encode(content)
        if len(tokens) > max_tokens:
            truncated = TOKENIZER.decode(tokens[:max_tokens])
            return _ensure_csv_header(truncated, list(df.columns)), max_tokens
        return _ensure_csv_header(content, list(df.columns)), len(tokens)
    except Exception as e:
        return f"Error reading CSV: {str(e)}", 0


def _serialize_csv_with_header(df: pd.DataFrame) -> str:
    return df.to_csv(index=False)


def _ensure_csv_header(content: str, columns: list[str]) -> str:
    header = ",".join(str(col) for col in columns)
    if not header:
        return content

    text = str(content or "")
    lines = text.splitlines()
    if not lines:
        return header + "\n"

    first_line = lines[0].strip()
    if first_line == header:
        return text if text.endswith("\n") else text + "\n"

    filtered_lines = [line for line in lines if line.strip() and line.strip() != header]
    merged = "\n".join([header, *filtered_lines])
    return merged if merged.endswith("\n") else merged + "\n"


def _reduce_history_prompt_context(system_instruction: str, ratio: float = 0.8) -> str:
    marker = "Here is the preview/content of the history data:\n\n"
    if marker not in system_instruction:
        return system_instruction

    prefix, table_content = system_instruction.split(marker, 1)
    text = str(table_content or "")
    lines = text.splitlines()
    if not lines:
        return system_instruction

    header = lines[0]
    body = "\n".join(lines[1:])
    body_tokens = TOKENIZER.encode(body)
    if not body_tokens:
        return system_instruction

    target_len = max(1, int(len(body_tokens) * ratio))
    reduced_body = TOKENIZER.decode(body_tokens[:target_len]).strip("\n")
    reduced_table = header
    if reduced_body:
        reduced_table += "\n" + reduced_body
    if not reduced_table.endswith("\n"):
        reduced_table += "\n"

    return prefix + marker + reduced_table


def _estimate_target_rows(filepath: str, max_tokens: int) -> int:
    total_rows = 0
    try:
        result = subprocess.run(['wc', '-l', filepath], capture_output=True, text=True)
        if result.returncode == 0:
            total_rows = int(result.stdout.split()[0])
    except Exception:
        total_rows = 0

    if total_rows == 0:
        try:
            file_size = os.path.getsize(filepath)
            total_rows = max(1, file_size // 100)
        except Exception:
            total_rows = 1

    estimated_target_rows = (max_tokens * 4) // 150
    estimated_target_rows = max(50, min(estimated_target_rows, 3000))
    return total_rows, estimated_target_rows


def _read_csv_with_strategy(filepath: str, max_tokens: int, strategy: str) -> tuple:
    file_basename = os.path.basename(filepath)
    total_rows, estimated_target_rows = _estimate_target_rows(filepath, max_tokens)
    random_state = TRUNCATION_RANDOM_SEED

    if strategy == "head":
        nrows = max(50, min(estimated_target_rows, total_rows if total_rows > 0 else estimated_target_rows))
        df = pd.read_csv(
            filepath,
            nrows=nrows,
            encoding='utf-8',
            encoding_errors='replace',
            on_bad_lines='skip',
            low_memory=False,
        )
        return df, total_rows

    if strategy == "random":
        is_small_file = total_rows < estimated_target_rows * 2
        if is_small_file:
            df = pd.read_csv(
                filepath,
                encoding='utf-8',
                encoding_errors='replace',
                on_bad_lines='skip',
                low_memory=False,
            )
            df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
            return df, total_rows

        frac = min(1.0, (estimated_target_rows / max(total_rows, 1)) * 1.5)
        chunk_size = 50000
        chunk_list = []
        with pd.read_csv(
            filepath,
            chunksize=chunk_size,
            encoding='utf-8',
            encoding_errors='replace',
            on_bad_lines='skip',
            low_memory=False,
        ) as reader:
            for chunk_idx, chunk in enumerate(reader):
                if chunk.empty:
                    continue
                chunk_seed = random_state + chunk_idx
                take = max(1, min(len(chunk), int(len(chunk) * frac)))
                sampled = chunk.sample(n=take, random_state=chunk_seed)
                chunk_list.append(sampled)
        if not chunk_list:
            return None, total_rows
        df = pd.concat(chunk_list, ignore_index=True)
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        return df, total_rows

    if strategy == "stratified":
        chunk_size = 50000
        chunk_list = []
        rows_per_chunk_hint = max(1, estimated_target_rows // max(1, (total_rows // chunk_size) + 1))
        with pd.read_csv(
            filepath,
            chunksize=chunk_size,
            encoding='utf-8',
            encoding_errors='replace',
            on_bad_lines='skip',
            low_memory=False,
        ) as reader:
            for chunk_idx, chunk in enumerate(reader):
                if chunk.empty:
                    continue
                if len(chunk) <= rows_per_chunk_hint:
                    sampled = chunk.copy()
                else:
                    sample_idx = sorted(set(
                        int(i) for i in np.linspace(0, len(chunk) - 1, num=rows_per_chunk_hint)
                    ))
                    sampled = chunk.iloc[sample_idx].copy()
                sampled["_chunk_order"] = chunk_idx
                chunk_list.append(sampled)
        if not chunk_list:
            return None, total_rows
        df = pd.concat(chunk_list, ignore_index=True)
        if "_chunk_order" in df.columns:
            df = df.sort_values("_chunk_order").drop(columns=["_chunk_order"]).reset_index(drop=True)
        return df, total_rows

    raise ValueError(f"Unsupported truncation strategy: {strategy}")

def get_file_content_truncated_by_tokens(filepath: str, max_tokens: int, strategy: str | None = None) -> tuple:
    """获取文件内容（带Token截断），返回 (content, token_count)"""
    if not os.path.exists(filepath):
        return "File not found", 0

    strategy = strategy or TRUNCATION_STRATEGY
    file_basename = os.path.basename(filepath)
    try:
        df, _ = _read_csv_with_strategy(filepath, max_tokens, strategy)
        if df is None or df.empty:
            return "Empty file", 0

        columns = list(df.columns)
        content = _serialize_csv_with_header(df)
        del df
        gc.collect()

        estimated_char_limit = max_tokens * 5
        content_to_encode = content[:estimated_char_limit] if len(content) > estimated_char_limit else content
        tokens = TOKENIZER.encode(content_to_encode)
        num_tokens = len(tokens)

        if num_tokens > max_tokens:
            truncated = TOKENIZER.decode(tokens[:max_tokens])
            return _ensure_csv_header(truncated, columns), max_tokens
        return _ensure_csv_header(content_to_encode, columns), num_tokens
    except Exception as e:
        print(f"[{file_basename}] Read error with strategy={strategy}: {e}, fallback to head.")
        return _fallback_read_head(filepath, max_tokens)

# ================= 任务处理 =================

def process_inference_task(task_args):
    """统一的任务处理函数"""
    task_start_time = time.time()
    
    task_type = task_args["task_type"]
    file_path = task_args["file_path"]
    mode = task_args["mode"]
    truncation_strategy = task_args.get("truncation_strategy", TRUNCATION_STRATEGY)
    skip_existing = task_args.get("skip_existing", False)  # 获取跳过开关
    
    file_name = os.path.basename(file_path)
    # dir_name = os.path.dirname(file_path)
    base_name = os.path.splitext(file_name)[0]

    out_info = task_args["output_info"]
    target_dir = os.path.join(
        out_info["root_dir"], 
        out_info["model_name"], 
        mode,
        out_info["task_type"], 
        out_info["dataset_rel_path"]
    )
    
    os.makedirs(target_dir, exist_ok=True)
    
    base_name = os.path.splitext(file_name)[0]

    output_name = f"{base_name}_{MODEL_ID.replace('/', '_')}_{mode}.json"
    # output_path = os.path.join(dir_name, output_name)
    output_path = os.path.join(target_dir, output_name)
    # === 新增：定义 CSV 输出路径 ===
    csv_output_name = output_name.replace(".json", ".csv")
    csv_output_path = os.path.join(target_dir, csv_output_name)
    # 如果已经存在旧的 csv，先删除，防止误判
    if os.path.exists(csv_output_path):
        try:
            os.remove(csv_output_path)
        except:
            pass
    # ============================

    if skip_existing and os.path.exists(output_path):
    # 可选：检查文件大小，防止跳过生成失败的空文件
        if os.path.getsize(output_path) > 0:
            print(f"[Skip] {file_name} already exists.") # 只有调试时才打印，防止刷屏
            return 0.0 # 直接返回耗时 0
    
    log_file_name = f"{base_name}_{MODEL_ID.replace('/', '_')}_{mode}.log"
    # [修改] 使用 target_dir
    log_path = os.path.join(target_dir, log_file_name)

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"Processing: {file_name}\nTime: {pd.Timestamp.now()}\n{'-'*50}\n")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        ground_truth = data.get("ground_truth", "unknown")
        inner_task_type = ground_truth.get("task_sub_type", "unknown")
        
        if task_type in ["B1", "B2", "B3"]:
            final_res, metrics = run_inference_single_csv(
                data.get("query", ""),
                task_args["table_ctx"],
                task_args["abs_history_path"],
                mode,
                log_path
            )
        else:
            mount_files = {
                "history.csv": task_args["abs_history_path"],
                "current.csv": task_args["abs_current_path"]
            }
            # === 修改：传入 inner_task_type 和 csv_output_path ===
            final_res, metrics = run_inference_dual_csv(
                data.get("query", ""),
                task_args["history_ctx"],
                task_args["current_ctx"],
                mount_files,
                mode,
                log_path,
                inner_task_type=inner_task_type,
                csv_output_path=csv_output_path if mode == "with_tool" else None
            )
            # =================================================

        # ==================== 修改开始 ====================
        # 检查推理结果是否包含代码中定义的错误标识
        # 你的 run_inference 函数在 except 时会返回 "Error...", "System Error...", "Critical API Error..."
        error_prefixes = ("Error", "System Error", "Critical API Error")
        
        if final_res.startswith(error_prefixes):
            print(f"[Skip Save] Inference error for {file_name}: {final_res[:100]}...")
            # 遇到错误直接返回，不保存 JSON
            return f"[Inference Failed] {file_name}"
        # ==================== 修改结束 ====================

        task_end_time = time.time()
        total_duration = task_end_time - task_start_time
        
        new_data = data.copy()
        new_data["response"] = final_res
        new_data["context_truncation_strategy"] = truncation_strategy
        if mode == "with_tool" and task_type == "B4":
            new_data["csv_success"] = os.path.exists(csv_output_path)
        else:
            new_data["csv_success"] = "N/A"
        new_data["performance_metrics"] = {
            "total_duration_seconds": round(total_duration, 2),
            "step_count": metrics["total_steps"],
            "code_execution_total": metrics["code_total_count"],
            "code_execution_success": metrics["code_success_count"],
            "step_time_details": metrics["step_details"]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)
            
        return total_duration
    except Exception as e:
        return f"[Error] {file_name}: {str(e)}"

def process_preprocessing_task(ds_folder, task_config):
    """预处理任务"""
    tasks = []
    task_type = task_config["task_type"]
    mode = task_config["mode"]
    has_current = task_config["has_current"]
    file_filter = task_config["file_filter"]
    max_file_tokens = task_config["context_size"] - RESERVED_TOKENS
    truncation_strategy = task_config.get("truncation_strategy", TRUNCATION_STRATEGY)
    skip_existing_flag = task_config.get("skip_existing", False)

    dataset_rel_path = os.path.relpath(ds_folder, task_config["base_dir"])
    
    try:
        history_csv_path = os.path.join(ds_folder, "history.csv")
        if not os.path.exists(history_csv_path):
            return []
        
        abs_history_path = os.path.abspath(history_csv_path)
        
        # --- 修改开始 ---
        # 逻辑调整：先判断是否需要处理 current.csv，再决定 history.csv 的读取策略
        current_ctx = None
        abs_current_path = None
        
        if has_current:
            current_csv_path = os.path.join(ds_folder, "current.csv")
            if not os.path.exists(current_csv_path):
                return []
            abs_current_path = os.path.abspath(current_csv_path)
            
            if mode == "no_tool":
                # B4 no_tool: 优先读取 current，剩余空间给 history
                current_ctx, current_tokens = get_file_content_truncated_by_tokens(current_csv_path, max_file_tokens, strategy=truncation_strategy)
                remaining_tokens = max(1000, max_file_tokens - current_tokens)
                history_ctx, _ = get_file_content_truncated_by_tokens(history_csv_path, remaining_tokens, strategy=truncation_strategy)
                gc.collect()
            else:
                # B4 with_tool: 仅读取列名
                current_ctx = get_file_columns(current_csv_path)
                history_ctx = get_file_columns(history_csv_path)
        else:
            # B1/B2/B3: 仅处理 history
            if mode == "no_tool":
                history_ctx, _ = get_file_content_truncated_by_tokens(history_csv_path, max_file_tokens, strategy=truncation_strategy)
            else:
                history_ctx = get_file_columns(history_csv_path)
        # --- 修改结束 ---

        for file_name in os.listdir(ds_folder):
            if file_filter(file_name):
                file_path = os.path.join(ds_folder, file_name)
                
                task_args = {
                    "task_type": task_type,
                    "file_path": file_path,
                    "mode": mode,
                    "abs_history_path": abs_history_path,
                    "skip_existing": skip_existing_flag,
                    "truncation_strategy": truncation_strategy,
                    "output_info": {
                        "root_dir": task_config["output_dir"],
                        "model_name": task_config["model_name_simple"],
                        "task_type": task_type,
                        "dataset_rel_path": dataset_rel_path
                    }
                }
                
                if has_current:
                    task_args["history_ctx"] = history_ctx
                    task_args["current_ctx"] = current_ctx
                    task_args["abs_current_path"] = abs_current_path
                else:
                    task_args["table_ctx"] = history_ctx
                
                tasks.append(task_args)
        
        return tasks
    except Exception as e:
        return []

def find_dataset_folders(base_dir, has_current=False):
    """扫描数据集文件夹"""
    dataset_folders = []
    for root, dirs, files in os.walk(base_dir):
        if "history.csv" in files:
            if has_current:
                if "current.csv" in files:
                    dataset_folders.append(root)
            else:
                dataset_folders.append(root)
    return dataset_folders

def run_single_task(task_name, workers, model_name, base_dir_override=None, mode_override=None, output_dir="outputs", skip_existing=False, truncation_strategy=None, max_files=None):
    """运行单个任务"""
    task_config = TASK_CONFIGS[task_name].copy()
    task_config["task_type"] = task_name
    task_config["output_dir"] = output_dir
    task_config["model_name_simple"] = model_name  # 保存简短的模型名(如 qwen)用于文件夹命名
    task_config["skip_existing"] = skip_existing
    task_config["truncation_strategy"] = truncation_strategy or TRUNCATION_STRATEGY
    
    if base_dir_override:
        task_config["base_dir"] = base_dir_override
    if mode_override:
        task_config["mode"] = mode_override
    
    global MAX_WORKERS
    MAX_WORKERS = workers
    
    print(f"\n{'='*60}")
    print(f"=== Task: {task_name} | Model: {model_name} | Truncation: {task_config.get('truncation_strategy', TRUNCATION_STRATEGY)} ===")
    print(f"{'='*60}\n")
    
    dataset_folders = find_dataset_folders(task_config["base_dir"], task_config["has_current"])
    print(f"Found {len(dataset_folders)} dataset folders")
    
    all_inference_tasks = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_preprocessing_task, folder, task_config): folder for folder in dataset_folders}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
            result = future.result()
            if isinstance(result, list):
                all_inference_tasks.extend(result)
    
    print(f"Generated {len(all_inference_tasks)} inference tasks\n")
    if max_files is not None:
        all_inference_tasks = all_inference_tasks[:max(0, int(max_files))]
        print(f"Limited to {len(all_inference_tasks)} inference tasks for this run\n")
    
    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_inference_task, task): task for task in all_inference_tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Inference"):
            try:
                result = future.result()
                if isinstance(result, float):
                    completed += 1
            except Exception as e:
                print(f"Error: {e}")
    
    print(f"\nCompleted: {completed}/{len(all_inference_tasks)}")
    return {"completed": completed, "total": len(all_inference_tasks)}

def main():
    parser = argparse.ArgumentParser(description="Multi-Model Inference Script")
    parser.add_argument("--task", type=str, required=True, choices=["B1", "B2", "B3", "B4", "ALL"])
    parser.add_argument("--model", type=str, required=True, 
                       choices=["gpt", "chatgpt52", "deepseek", "qwen", "claude", "gemini", "deepseek_reasoner", "qwen_thinking"],
                       help="Model to use")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs", help="Root directory for outputs")
    parser.add_argument("--mode", type=str, default=None, choices=["no_tool", "with_tool"])
    parser.add_argument(
        "--truncation-strategy",
        type=str,
        default="random",
        choices=["head", "random", "stratified"],
        help="Row selection strategy for no_tool table truncation experiments",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip if output JSON already exists")
    parser.add_argument("--max-files", type=int, default=None, help="Limit the number of inference files for smoke tests")
    
    args = parser.parse_args()
    
    global TRUNCATION_STRATEGY
    TRUNCATION_STRATEGY = args.truncation_strategy

    # 初始化模型
    initialize_model(args.model)
    
    if args.task == "ALL":
        for task_name in ["B1", "B2", "B3", "B4"]:
            run_single_task(
                task_name,
                args.workers,
                args.model,
                output_dir=args.output_dir,
                skip_existing=args.skip_existing,
                truncation_strategy=args.truncation_strategy,
                max_files=args.max_files,
            )
    else:
        run_single_task(
            args.task,
            args.workers,
            args.model,
            args.base_dir,
            args.mode,
            output_dir=args.output_dir,
            skip_existing=args.skip_existing,
            truncation_strategy=args.truncation_strategy,
            max_files=args.max_files,
        )

try:
    TOKENIZER = tiktoken.get_encoding("cl100k_base")
except:
    try:
        TOKENIZER = tiktoken.get_encoding("p50k_base")
    except:
        class SimpleTokenizer:
            @staticmethod
            def encode(text):
                if not text:
                    return []
                return [text[i:i+4] for i in range(0, len(text), 4)]

            @staticmethod
            def decode(tokens):
                return "".join(tokens)

        TOKENIZER = SimpleTokenizer()

if __name__ == "__main__":
    main()

# python infer_all_v2.py --task B2 --model deepseek --workers 2 --mode with_tool
