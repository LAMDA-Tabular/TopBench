# Release Audit Notes

This clean release tree was rebuilt from the research workspace instead of reusing the previous `release_staging/TopBench` directory directly.

Excluded from the public release:

- `catboost_info/`
- `__pycache__/`
- `.Rhistory`
- `evaluation.log` and other local logs
- smoke-test outputs
- full benchmark data
- full model outputs
- rebuttal-only scripts and figures
- AIDE baseline code
- hard-coded API keys

The legacy scripts in the research workspace remain frozen. Public entry points are under `scripts/`, and shared code is under `src/topbench/`.
