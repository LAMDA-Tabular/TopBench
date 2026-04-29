from __future__ import annotations

import re
from typing import Any

from topbench.evaluation.metric_utils import normalize_text


def extract_numbers(text: str) -> list[float]:
    values: list[float] = []
    for match in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text or ""):
        try:
            values.append(float(match.replace(",", "")))
        except ValueError:
            continue
    return values


def value_is_supported_by_response(value: Any, response: str) -> bool:
    """Check whether an extracted value/category is explicitly present in the response."""
    if value is None:
        return False
    if normalize_text(value) and normalize_text(value) in normalize_text(response):
        return True
    try:
        target = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return False
    return any(abs(candidate - target) <= max(1e-8, 0.01 * max(abs(target), 1.0)) for candidate in extract_numbers(response))
