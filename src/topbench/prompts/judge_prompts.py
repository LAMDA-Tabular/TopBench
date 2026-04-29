from __future__ import annotations

from typing import Dict, List, Optional


class PromptBuilder:
    """Prompt templates extracted from eval_b2_v1.py."""

    COMMON_FLAWS_CHECKLIST = """
    **STEP 1: FLAW DETECTION (CHECK FOR THE "SIX SINS")**
    1. **Self-Contradiction**: Text conflicts with prediction or ignores cited features.
    2. **Tautology**: Circular logic (explaining the result with the result itself).
    3. **Repetitive**: Restating the same point without new info.
    4. **Vacuous Fluff**: Empty fillers ("As an AI...", "Complex analysis") with no substance.
    5. **False Causality**: Linking irrelevant inputs (e.g., IDs) to outputs.
    6. **Over-Hedging**: Refusing to conclude ("Could be A or B").
    """

    SYMBOL_RULE = """
    **SYMBOL INTERPRETATION RULE**:
    - **Context determines Sign**: Symbols like `−` (unicode minus), `-`, or `~` can be ambiguous.
    - If the text uses `−$912` to mean "**slightly less than $912**" (approximate), you MUST extract **912** (Positive).
    """

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

    MISSING_PREDICTION_RULE = """
    **SILENCE = NULL (DO NOT INFER)**:
    - If the model discusses a Scenario (e.g., analyzes its risk) but **DOES NOT explicitly state** the value for the target column, you MUST set `predicted_value`/`predicted_category` to `null`.
    - **Example**: If text says "Scenario 1 is Class A, Scenario 2 is bad", extract "Class A" for S1, but `null` for S2.
    - **STRICT PROHIBITION**: Do not copy the prediction from Scenario 1 to Scenario 2 unless the text explicitly says "Scenario 2 is the same".
    """

    COMPARISON_RULE = """
    **HANDLING COMPARISONS (CRITICAL)**:
    - If the text says "Option A is better than Option B":
      - For Option A: Quote "Option A is better than Option B".
      - For Option B: **ALSO Quote "Option A is better than Option B"**.
    - **DO NOT MODIFY THE TEXT**: Never swap names/IDs to make the sentence fit.
    - **Example**: If text says "ID 37651 has higher risk than ID 10436", do NOT write "ID 10436 has higher risk" as a quote. Quote the original sentence exactly.
    """

    @staticmethod
    def get_logic_criteria(mode: str = "no_tool") -> str:
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

    @staticmethod
    def build_regression_prompt(
        query: str,
        gt_str: str,
        response: str,
        history_snippet: str,
        dataset_meta_str: str,
        mode: str = "no_tool",
    ) -> str:
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
    def build_classification_prompt(
        query: str,
        gt_str: str,
        response: str,
        history_snippet: str,
        dataset_meta_str: str,
        target_col: str,
        valid_cats: List,
        mode: str = "no_tool",
    ) -> str:
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
    def build_choice_prompt(
        query: str,
        scenarios_info: List[Dict],
        gt_decision: str,
        response: str,
        dataset_meta_str: str,
        target_col: str,
        task_type: str,
        valid_cats: Optional[List] = None,
        mode: str = "no_tool",
    ) -> str:
        scenarios_desc = []
        for item in scenarios_info:
            sid = item.get("scenario_id")
            feats = item.get("features", {})
            desc_tokens = [f"scenario_id: {sid}"]
            for k in feats:
                if k == target_col:
                    continue
                desc_tokens.append(f"{k}: {feats[k]}")
            scenarios_desc.append(" | ".join(desc_tokens))
        scenarios_block = "\n".join(scenarios_desc)
        logic_block = PromptBuilder.get_logic_criteria(mode)

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
        else:
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
    def build_whatif_prompt(
        query: str,
        scenarios_info: List[Dict],
        gt_trend: str,
        response: str,
        dataset_meta_str: str,
        target_col: str,
        task_type: str,
        valid_cats: Optional[List] = None,
        mode: str = "no_tool",
    ) -> str:
        scenarios_desc = []
        for item in scenarios_info:
            sid = item.get("scenario_id")
            feats = item.get("features", {})
            desc_tokens = [f"Scenario ID: {sid}"]
            for k in feats:
                if k == target_col:
                    continue
                desc_tokens.append(f"{k}: {feats[k]}")
            scenarios_desc.append(" | ".join(desc_tokens))
        scenarios_block = "\n".join(scenarios_desc)
        logic_block = PromptBuilder.get_logic_criteria(mode)

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
        else:
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


def build_regression_prompt(
    *,
    query: str,
    gt_str: str,
    response: str,
    history_snippet: str,
    dataset_meta_str: str,
    mode: str = "no_tool",
) -> str:
    return PromptBuilder.build_regression_prompt(
        query=query,
        gt_str=gt_str,
        response=response,
        history_snippet=history_snippet,
        dataset_meta_str=dataset_meta_str,
        mode=mode,
    )


def build_classification_prompt(
    *,
    query: str,
    gt_str: str,
    response: str,
    history_snippet: str,
    dataset_meta_str: str,
    target_col: str,
    valid_cats: List,
    mode: str = "no_tool",
) -> str:
    return PromptBuilder.build_classification_prompt(
        query=query,
        gt_str=gt_str,
        response=response,
        history_snippet=history_snippet,
        dataset_meta_str=dataset_meta_str,
        target_col=target_col,
        valid_cats=valid_cats,
        mode=mode,
    )


def build_choice_prompt(
    *,
    query: str,
    scenarios_info: List[Dict],
    gt_decision: str,
    response: str,
    dataset_meta_str: str,
    target_col: str,
    task_type: str,
    valid_cats: Optional[List] = None,
    mode: str = "no_tool",
) -> str:
    return PromptBuilder.build_choice_prompt(
        query=query,
        scenarios_info=scenarios_info,
        gt_decision=gt_decision,
        response=response,
        dataset_meta_str=dataset_meta_str,
        target_col=target_col,
        task_type=task_type,
        valid_cats=valid_cats,
        mode=mode,
    )


def build_whatif_prompt(
    *,
    query: str,
    scenarios_info: List[Dict],
    gt_trend: str,
    response: str,
    dataset_meta_str: str,
    target_col: str,
    task_type: str,
    valid_cats: Optional[List] = None,
    mode: str = "no_tool",
) -> str:
    return PromptBuilder.build_whatif_prompt(
        query=query,
        scenarios_info=scenarios_info,
        gt_trend=gt_trend,
        response=response,
        dataset_meta_str=dataset_meta_str,
        target_col=target_col,
        task_type=task_type,
        valid_cats=valid_cats,
        mode=mode,
    )
