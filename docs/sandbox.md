# Sandbox Setup

TopBench uses a sandbox for the agentic workflow because models may generate and execute Python code over CSV files.

Build the image:

```bash
docker build -f docker/Dockerfile.sandbox -t topbench-sandbox:latest .
```

Run the health check:

```bash
python scripts/healthcheck_sandbox.py --image topbench-sandbox:latest
```

The legacy-compatible inference script uses `topbench-sandbox:latest` unless `TOPBENCH_SANDBOX_IMAGE` is set.

Run a sampled legacy-compatible agentic workflow:

```bash
python scripts/run_legacy_inference.py \
  --data-root data \
  --output-root outputs \
  --model deepseek \
  --tasks ranking_and_filtering \
  --modes agentic_workflow \
  --max-files 1
```

Recommended restrictions:

- Run containers with `--network none` for generated code execution.
- Mount only the dataset folder and output folder needed by the current run.
- Do not mount API keys or the repository root into the execution container unless required.
- Treat generated code as untrusted.

On macOS or Colima, Docker can only mount paths inside configured file-sharing roots. `scripts/run_legacy_inference.py` automatically sets `TOPBENCH_SANDBOX_TMPDIR` under the output root so temporary code files remain mountable.

The lightweight `scripts/run_agentic_workflow.py` runner is a clean public scaffold. Use `scripts/run_legacy_inference.py --modes agentic_workflow` when you need the paper-era output contract. Text-based inference and predict-only baselines do not require the sandbox.
