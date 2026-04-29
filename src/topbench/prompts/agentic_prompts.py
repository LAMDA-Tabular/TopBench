from __future__ import annotations

from topbench.prompts.prompt_templates import render_template

DEFAULT_MAX_ITERATIONS = 10

# Extracted from infer_all_v3.py.

SINGLE_CSV_TOOL_SYSTEM_PROMPT = """The history data file is located at: history.csv
(Note: The file is mounted in your environment, use 'history.csv' directly)
Here are the columns of the file:
[$column_preview]

IMPORTANT: You MUST use the 'CodeRunner' tool to read the file to inspect the data content.
You need to give the answer within $max_iterations rounds."""

DUAL_CSV_TOOL_SYSTEM_PROMPT = """You have access to two csv files in your environment:
1. 'history.csv'
   - Columns: $history_columns
2. 'current.csv'
   - Columns: $current_columns

Note: These files are mounted, use their filenames directly.
The data provided above are ONLY column names. DO NOT hallucinate data rows.
You MUST use the CodeRunner tool to read the files (e.g., pd.read_csv) to inspect the actual data content.

CRITICAL REQUIREMENT:
1. You MUST process the data and save the final results into a file named 'result.csv'.
2. The 'result.csv' MUST contain the exact columns matching history.csv format.
3. Do not just print the result, you must save it to 'result.csv' using pandas to_csv().$prompt_extras
You need to give the answer within $max_iterations rounds."""

REGRESSION_RANKING_WARNING = (
    "\nWARNING: This is a REGRESSION task. When outputting the list, strictly verify the "
    "Top-K order based on your estimated values (Descending/Ascending as required)."
)

OPENAI_FINAL_TOOL_STOP_INSTRUCTION = (
    "You have reached the maximum number of steps allowed. "
    "Do NOT use any more tools (CodeRunner/PipInstaller). "
    "Based on the information you have gathered so far, please summarize your findings "
    "and provide a final answer to the user's original query."
)

GEMINI_FINAL_TOOL_STOP_INSTRUCTION = (
    "You have reached the maximum number of steps allowed. "
    "Do NOT use any more tools. "
    "Based on the information you have gathered so far, please summarize your findings "
    "and provide a final answer to the user's original query."
)


def build_single_csv_tool_system_prompt(
    *,
    column_preview: str,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> str:
    return render_template(
        SINGLE_CSV_TOOL_SYSTEM_PROMPT,
        {
            "column_preview": column_preview,
            "max_iterations": max_iterations,
        },
    )


def build_dual_csv_tool_system_prompt(
    *,
    history_columns: str,
    current_columns: str,
    inner_task_type: str | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> str:
    prompt_extras = ""
    if inner_task_type and str(inner_task_type).lower() == "regression":
        prompt_extras = REGRESSION_RANKING_WARNING
    return render_template(
        DUAL_CSV_TOOL_SYSTEM_PROMPT,
        {
            "history_columns": history_columns,
            "current_columns": current_columns,
            "prompt_extras": prompt_extras,
            "max_iterations": max_iterations,
        },
    )


def build_openai_final_instruction() -> str:
    return OPENAI_FINAL_TOOL_STOP_INSTRUCTION


def build_gemini_final_instruction() -> str:
    return GEMINI_FINAL_TOOL_STOP_INSTRUCTION


def build_agentic_system_prompt(
    *,
    dataset_dir: str,
    output_dir: str,
    column_preview: str = "",
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> str:
    # Compatibility wrapper for the current public runner interface.
    del dataset_dir, output_dir
    return build_single_csv_tool_system_prompt(
        column_preview=column_preview,
        max_iterations=max_iterations,
    )
