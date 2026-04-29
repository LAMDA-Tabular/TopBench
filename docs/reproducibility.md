# Reproducibility Notes

TopBench separates three stages:

1. Inference creates raw model outputs.
2. Extraction converts free-form answers into structured values or labels.
3. Deterministic metrics compute final scores.

For `single_point_prediction`, `decision_making`, and `treatment_effect_analysis`, the LLM judge is used only as an extractor or formatter. It is not the final scorer. For `ranking_and_filtering`, evaluation is file-based. The compatibility replay uses `eval_b4_v3.py`, which is the evaluator that matches the current paper summary files; `eval_v4_v2.py` is kept as an older reference script.

## Recommended Public Reproduction Path

1. Prepare the canonical public dataset layout under `data/`.
2. Run sampled or full legacy-compatible inference with `scripts/run_legacy_inference.py` when you need the paper-era output schema.
3. Evaluate frozen outputs with `scripts/reproduce_paper_scores.py`.
4. Replay summary files with `scripts/reproduce_reasoning_summaries.py` and `scripts/reproduce_structured_summary.py`.

## Important Boundary

Exact live inference reproduction is not guaranteed because hosted APIs may change over time. The public repository therefore treats frozen-output evaluation as the authoritative score-reproduction path.

Generated reproduction outputs, logs, temporary summaries, and any comparisons against private historical workspaces are local verification artifacts. They should not be committed to GitHub.
