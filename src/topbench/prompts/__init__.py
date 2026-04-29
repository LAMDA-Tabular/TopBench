"""Prompt templates extracted from the original TopBench infer/eval scripts."""

from topbench.prompts.agentic_prompts import (
    DEFAULT_MAX_ITERATIONS,
    GEMINI_FINAL_TOOL_STOP_INSTRUCTION,
    OPENAI_FINAL_TOOL_STOP_INSTRUCTION,
    build_agentic_system_prompt,
    build_dual_csv_tool_system_prompt,
    build_gemini_final_instruction,
    build_openai_final_instruction,
    build_single_csv_tool_system_prompt,
)
from topbench.prompts.judge_prompts import (
    PromptBuilder,
    build_choice_prompt,
    build_classification_prompt,
    build_regression_prompt,
    build_whatif_prompt,
)
from topbench.prompts.reasoning_prompts import (
    build_dual_csv_no_tool_prompt_parts,
    build_single_csv_no_tool_prompt_parts,
    build_text_reasoning_prompt,
)

__all__ = [
    "DEFAULT_MAX_ITERATIONS",
    "GEMINI_FINAL_TOOL_STOP_INSTRUCTION",
    "OPENAI_FINAL_TOOL_STOP_INSTRUCTION",
    "PromptBuilder",
    "build_agentic_system_prompt",
    "build_choice_prompt",
    "build_classification_prompt",
    "build_dual_csv_no_tool_prompt_parts",
    "build_dual_csv_tool_system_prompt",
    "build_gemini_final_instruction",
    "build_openai_final_instruction",
    "build_regression_prompt",
    "build_single_csv_no_tool_prompt_parts",
    "build_single_csv_tool_system_prompt",
    "build_text_reasoning_prompt",
    "build_whatif_prompt",
]
