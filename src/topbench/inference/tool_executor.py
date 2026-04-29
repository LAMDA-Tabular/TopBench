from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SandboxCommand:
    image: str
    workdir: Path
    command: str


def run_in_sandbox(command: SandboxCommand, *, timeout: int = 300) -> subprocess.CompletedProcess:
    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "-v",
        f"{command.workdir.resolve()}:/workspace",
        "-w",
        "/workspace",
        command.image,
        "bash",
        "-lc",
        command.command,
    ]
    return subprocess.run(docker_cmd, capture_output=True, text=True, timeout=timeout, check=False)
