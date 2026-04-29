from __future__ import annotations

from string import Template
from typing import Any, Dict


def render_template(template: str, values: Dict[str, Any]) -> str:
    safe_values = {key: "" if value is None else str(value) for key, value in values.items()}
    return Template(template).safe_substitute(safe_values)
