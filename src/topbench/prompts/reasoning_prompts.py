from __future__ import annotations

from topbench.prompts.prompt_templates import render_template

# Extracted from infer_all_v3.py. The original script uses separate system/user
# messages; the public text runner currently consumes a single prompt string, so
# we expose both the exact prompt parts and a concatenated compatibility wrapper.

SINGLE_CSV_NO_TOOL_SYSTEM_PROMPT = """Here is the preview/content of the history data:

$history_table"""

SINGLE_CSV_NO_TOOL_USER_PROMPT = """$query"""

DUAL_CSV_NO_TOOL_SYSTEM_PROMPT = """Here is the preview/content of the history data:

$history_table"""

DUAL_CSV_NO_TOOL_USER_PROMPT = """$query

$current_table"""


def build_single_csv_no_tool_prompt_parts(*, query: str, history_table: str) -> tuple[str, str]:
    system_prompt = render_template(
        SINGLE_CSV_NO_TOOL_SYSTEM_PROMPT,
        {"history_table": history_table},
    )
    user_prompt = render_template(
        SINGLE_CSV_NO_TOOL_USER_PROMPT,
        {"query": query},
    )
    return system_prompt, user_prompt


def build_dual_csv_no_tool_prompt_parts(
    *,
    query: str,
    history_table: str,
    current_table: str,
) -> tuple[str, str]:
    system_prompt = render_template(
        DUAL_CSV_NO_TOOL_SYSTEM_PROMPT,
        {"history_table": history_table},
    )
    user_prompt = render_template(
        DUAL_CSV_NO_TOOL_USER_PROMPT,
        {"query": query, "current_table": current_table},
    )
    return system_prompt, user_prompt


def build_text_reasoning_prompt(*, task_display_name: str, query: str, history_table: str) -> str:
    # Kept for the public runner's current single-string client interface.
    del task_display_name
    system_prompt, user_prompt = build_single_csv_no_tool_prompt_parts(
        query=query,
        history_table=history_table,
    )
    return f"{system_prompt}\n\n{user_prompt}"
