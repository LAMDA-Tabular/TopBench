from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from topbench.compatibility.legacy_data_view import prepare_legacy_data_view
from topbench.compatibility.legacy_name_mapping import to_legacy_mode, to_legacy_task_name
from topbench.io.path_resolver import resolve_existing_task_dir


def bundled_legacy_code_root() -> Path:
    return Path(__file__).resolve().parent / "legacy_scripts"


def run_legacy_inference(
    *,
    data_root: str | Path,
    output_root: str | Path,
    task_name: str,
    model_name: str,
    mode: str,
    workers: int = 1,
    python_bin: str = "python",
    max_files: int | None = None,
    skip_existing: bool = True,
    truncation_strategy: str | None = None,
    legacy_code_root: str | Path | None = None,
) -> subprocess.CompletedProcess:
    data_root = Path(data_root).resolve()
    output_root = Path(output_root).resolve()
    root = Path(legacy_code_root) if legacy_code_root is not None else bundled_legacy_code_root()
    script = root / "infer_all_v3.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing legacy inference runner: {script}")

    task_data_root = resolve_existing_task_dir(data_root, task_name)
    cmd = [
        python_bin,
        str(script),
        "--task",
        to_legacy_task_name(task_name),
        "--model",
        model_name,
        "--workers",
        str(workers),
        "--base-dir",
        str(task_data_root),
        "--output-dir",
        str(output_root),
        "--mode",
        to_legacy_mode(mode),
    ]
    if max_files is not None:
        cmd.extend(["--max-files", str(max_files)])
    if skip_existing:
        cmd.append("--skip-existing")
    if truncation_strategy:
        cmd.extend(["--truncation-strategy", truncation_strategy])

    env = os.environ.copy()
    env.setdefault("TOPBENCH_LEGACY_DATA_ROOT", str(data_root))
    sandbox_tmp = output_root / ".topbench_sandbox_tmp"
    sandbox_tmp.mkdir(parents=True, exist_ok=True)
    env.setdefault("TOPBENCH_SANDBOX_TMPDIR", str(sandbox_tmp))
    return subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, check=False, env=env)


def run_legacy_reasoning_evaluator(
    *,
    legacy_code_root: str | Path | None = None,
    inference_root: str | Path,
    data_root: str | Path | None = None,
    task_name: str,
    model_name: str,
    mode: str,
    workers: int = 8,
    python_bin: str = "python",
    skip_existing: bool = True,
) -> subprocess.CompletedProcess:
    inference_root = Path(inference_root).resolve()
    root = Path(legacy_code_root) if legacy_code_root is not None else bundled_legacy_code_root()
    script = root / "eval_b2_v1.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing legacy evaluator: {script}")

    with tempfile.TemporaryDirectory(prefix="topbench_legacy_eval_") as temp_dir:
        env = os.environ.copy()
        if data_root is not None:
            prepare_legacy_data_view(Path(data_root).resolve(), temp_dir, tasks=[task_name])
            env["TOPBENCH_LEGACY_DATA_ROOT"] = str(Path(temp_dir).resolve())

        cmd = [
            python_bin,
            str(script),
            "--inference_root",
            str(inference_root),
            "--benchmark",
            to_legacy_task_name(task_name),
            "--models",
            model_name,
            "--mode",
            to_legacy_mode(mode),
            "--workers",
            str(workers),
        ]
        if not skip_existing:
            cmd.append("--no-skip-existing-eval")
        return subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, check=False, env=env)


def run_legacy_structured_evaluator(
    *,
    legacy_code_root: str | Path | None = None,
    inference_root: str | Path,
    data_root: str | Path | None = None,
    model_name: str,
    mode: str,
    python_bin: str = "python",
) -> subprocess.CompletedProcess:
    inference_root = Path(inference_root).resolve()
    root = Path(legacy_code_root) if legacy_code_root is not None else bundled_legacy_code_root()
    script = root / "eval_b4_v3.py"
    if not script.exists():
        script = root / "eval_v4_v2.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing legacy structured evaluator: {script}")

    with tempfile.TemporaryDirectory(prefix="topbench_legacy_b4_eval_") as temp_dir:
        cmd = [
            python_bin,
            str(script),
            "--inference_root",
            str(inference_root),
            "--models",
            model_name,
            "--mode",
            to_legacy_mode(mode),
        ]
        env = os.environ.copy()
        if data_root is not None:
            prepared = prepare_legacy_data_view(Path(data_root).resolve(), temp_dir, tasks=["ranking_and_filtering"])
            cmd.extend(["--dataset_root", str(prepared["ranking_and_filtering"])])
            env["TOPBENCH_LEGACY_DATA_ROOT"] = str(Path(temp_dir).resolve())
        return subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, check=False, env=env)


def run_legacy_predict_only_baseline(
    *,
    data_root: str | Path,
    output_root: str | Path,
    task_names: list[str],
    model_name: str = "predict_only_ensemble",
    mode: str = "predict_only",
    python_bin: str = "python",
    skip_existing: bool = False,
    max_files: int | None = None,
    accuracy_first: bool = False,
    full_complex_models: bool = False,
    gpu: str = "off",
    max_train_rows: int | None = None,
    tabpfn_max_rows: int | None = None,
    tabstar_max_rows: int | None = None,
    tabstar_time_limit: int | None = None,
    legacy_code_root: str | Path | None = None,
) -> subprocess.CompletedProcess:
    data_root = Path(data_root).resolve()
    output_root = Path(output_root).resolve()
    root = Path(legacy_code_root) if legacy_code_root is not None else bundled_legacy_code_root()
    script = root / "predict_only_benchmark.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing legacy predict-only baseline: {script}")

    with tempfile.TemporaryDirectory(prefix="topbench_predict_only_legacy_") as temp_dir:
        prepare_legacy_data_view(data_root, temp_dir, tasks=task_names)
        cmd = [
            python_bin,
            str(script),
            "--source-root",
            temp_dir,
            "--output-root",
            str(output_root),
            "--model-name",
            model_name,
            "--mode",
            mode,
            "--benchmarks",
            *[to_legacy_task_name(task_name) for task_name in task_names],
            "--gpu",
            gpu,
        ]
        if skip_existing:
            cmd.append("--skip-existing")
        if accuracy_first:
            cmd.append("--accuracy-first")
        if full_complex_models:
            cmd.append("--full-complex-models")
        if max_files is not None:
            cmd.extend(["--max-files", str(max_files)])
        if max_train_rows is not None:
            cmd.extend(["--max-train-rows", str(max_train_rows)])
        if tabpfn_max_rows is not None:
            cmd.extend(["--tabpfn-max-rows", str(tabpfn_max_rows)])
        if tabstar_max_rows is not None:
            cmd.extend(["--tabstar-max-rows", str(tabstar_max_rows)])
        if tabstar_time_limit is not None:
            cmd.extend(["--tabstar-time-limit", str(tabstar_time_limit)])
        return subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, check=False)
