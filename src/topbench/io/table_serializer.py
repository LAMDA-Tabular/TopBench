from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import pandas as pd

SamplingStrategy = Literal["head", "random", "stratified"]


def estimate_total_rows(csv_path: str | Path) -> int:
    path = Path(csv_path)
    try:
        result = subprocess.run(["wc", "-l", str(path)], capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            return max(0, int(result.stdout.split()[0]) - 1)
    except Exception:
        pass
    return 0


def estimate_target_rows(max_tokens: int) -> int:
    estimated = (max_tokens * 4) // 150
    return max(50, min(estimated, 3000))


def _ensure_header(csv_text: str, columns: list[str]) -> str:
    header = ",".join(str(col) for col in columns)
    if not header:
        return csv_text
    lines = [line for line in str(csv_text or "").splitlines() if line.strip()]
    if lines and lines[0].strip() == header:
        return "\n".join(lines) + "\n"
    return "\n".join([header, *[line for line in lines if line.strip() != header]]) + "\n"


def sample_csv_rows(
    csv_path: str | Path,
    *,
    max_tokens: int,
    strategy: SamplingStrategy = "random",
    random_seed: int = 42,
) -> pd.DataFrame:
    path = Path(csv_path)
    total_rows = estimate_total_rows(path)
    target_rows = estimate_target_rows(max_tokens)

    if strategy == "head":
        return pd.read_csv(path, nrows=target_rows, encoding_errors="replace", on_bad_lines="skip")

    if strategy == "random":
        if total_rows and total_rows <= target_rows * 2:
            df = pd.read_csv(path, encoding_errors="replace", on_bad_lines="skip", low_memory=False)
            return df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
        chunks = []
        frac = min(1.0, (target_rows / max(total_rows, 1)) * 1.5)
        for idx, chunk in enumerate(
            pd.read_csv(path, chunksize=50000, encoding_errors="replace", on_bad_lines="skip", low_memory=False)
        ):
            if chunk.empty:
                continue
            take = max(1, min(len(chunk), int(len(chunk) * frac)))
            chunks.append(chunk.sample(n=take, random_state=random_seed + idx))
        if not chunks:
            return pd.DataFrame()
        return pd.concat(chunks, ignore_index=True).sample(frac=1.0, random_state=random_seed)

    if strategy == "stratified":
        chunks = []
        chunk_size = 50000
        rows_per_chunk = max(1, target_rows // max(1, (total_rows // chunk_size) + 1))
        for idx, chunk in enumerate(
            pd.read_csv(path, chunksize=chunk_size, encoding_errors="replace", on_bad_lines="skip", low_memory=False)
        ):
            if chunk.empty:
                continue
            sample_idx = sorted(set(int(i) for i in np.linspace(0, len(chunk) - 1, rows_per_chunk)))
            sampled = chunk.iloc[sample_idx].copy()
            sampled["_topbench_chunk_order"] = idx
            chunks.append(sampled)
        if not chunks:
            return pd.DataFrame()
        df = pd.concat(chunks, ignore_index=True)
        return df.sort_values("_topbench_chunk_order").drop(columns=["_topbench_chunk_order"])

    raise ValueError(f"Unsupported sampling strategy: {strategy}")


def serialize_csv_for_prompt(
    csv_path: str | Path,
    *,
    max_tokens: int,
    tokenizer=None,
    strategy: SamplingStrategy = "random",
    random_seed: int = 42,
) -> Tuple[str, int]:
    df = sample_csv_rows(csv_path, max_tokens=max_tokens, strategy=strategy, random_seed=random_seed)
    if df.empty:
        return "", 0

    text = _ensure_header(df.to_csv(index=False), list(df.columns))
    if tokenizer is None:
        return text, len(text)

    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text, len(tokens)
    truncated = tokenizer.decode(tokens[:max_tokens])
    return _ensure_header(truncated, list(df.columns)), max_tokens
