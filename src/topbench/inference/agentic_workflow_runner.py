from __future__ import annotations

from pathlib import Path

from topbench.io.dataset_loader import iter_dataset_folders, load_json
from topbench.io.output_writer import output_file_for_query, write_prediction_output
from topbench.prompts.agentic_prompts import build_agentic_system_prompt
from topbench.task_registry import canonicalize_task_name


def run_agentic_workflow(
    *,
    task_name: str,
    data_root: str | Path,
    output_root: str | Path,
    model_name: str,
    sandbox_image: str,
    max_files: int | None = None,
) -> int:
    # The public runner keeps sandbox orchestration explicit. Full ReAct/tool logic is
    # implemented in the compatibility path to preserve the paper's original behavior.
    task = canonicalize_task_name(task_name)
    count = 0
    for dataset in iter_dataset_folders(data_root, task):
        system_prompt = build_agentic_system_prompt(
            dataset_dir=str(dataset.path),
            output_dir=str(Path(output_root).resolve()),
        )
        for query_file in dataset.query_files:
            query = load_json(query_file)
            output_path = output_file_for_query(
                output_root,
                model_name=model_name,
                mode="agentic_workflow",
                task_name=task,
                query_file=query_file,
            )
            payload = dict(query)
            payload["model_name"] = model_name
            payload["mode"] = "agentic_workflow"
            payload["sandbox_image"] = sandbox_image
            payload["system_prompt"] = system_prompt
            payload["response"] = ""
            payload["note"] = "Use the legacy-compatible agentic runner for paper-exact execution."
            write_prediction_output(output_path, payload)
            count += 1
            if max_files is not None and count >= max_files:
                return count
    return count
