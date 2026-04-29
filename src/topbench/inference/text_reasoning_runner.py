from __future__ import annotations

from pathlib import Path

from topbench.inference.model_clients import ChatClient
from topbench.io.dataset_loader import iter_dataset_folders, load_json
from topbench.io.output_writer import output_file_for_query, write_prediction_output
from topbench.io.table_serializer import serialize_csv_for_prompt
from topbench.prompts.reasoning_prompts import build_text_reasoning_prompt
from topbench.task_registry import CANONICAL_TASKS, canonicalize_task_name


def run_text_reasoning(
    *,
    task_name: str,
    data_root: str | Path,
    output_root: str | Path,
    model_name: str,
    max_table_tokens: int = 24000,
    max_files: int | None = None,
) -> int:
    task = canonicalize_task_name(task_name)
    client = ChatClient(model_name)
    count = 0
    for dataset in iter_dataset_folders(data_root, task):
        history_table, _ = serialize_csv_for_prompt(dataset.history_csv, max_tokens=max_table_tokens)
        for query_file in dataset.query_files:
            query = load_json(query_file)
            prompt = build_text_reasoning_prompt(
                task_display_name=CANONICAL_TASKS[task].display_name,
                query=str(query.get("query", "")),
                history_table=history_table,
            )
            response = client.complete(prompt)
            output_path = output_file_for_query(
                output_root,
                model_name=model_name,
                mode="text_reasoning",
                task_name=task,
                query_file=query_file,
            )
            payload = dict(query)
            payload["response"] = response
            payload["model_name"] = model_name
            payload["mode"] = "text_reasoning"
            write_prediction_output(output_path, payload)
            count += 1
            if max_files is not None and count >= max_files:
                return count
    return count
