from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ChatModelConfig:
    provider: str
    model_id: str
    api_key_env: str
    base_url_env: Optional[str] = None
    default_base_url: Optional[str] = None


MODEL_CONFIGS = {
    "deepseek": ChatModelConfig(
        provider="openai_compatible",
        model_id="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
        base_url_env="DEEPSEEK_BASE_URL",
        default_base_url="https://api.deepseek.com",
    ),
    "gpt": ChatModelConfig(
        provider="openai",
        model_id="gpt-5.2",
        api_key_env="OPENAI_API_KEY",
    ),
    "qwen": ChatModelConfig(
        provider="openai_compatible",
        model_id="qwen3-235b-a22b-instruct-2507",
        api_key_env="QWEN_API_KEY",
        base_url_env="QWEN_BASE_URL",
    ),
}


class ChatClient:
    def __init__(self, model_name: str, *, model_id: str | None = None):
        try:
            self.config = MODEL_CONFIGS[model_name]
        except KeyError as exc:
            raise ValueError(f"Unknown model '{model_name}'. Available: {sorted(MODEL_CONFIGS)}") from exc
        self.model_name = model_name
        self.model_id = model_id or self.config.model_id

    def complete(self, prompt: str, *, temperature: float = 0.1, max_tokens: int = 4096) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the 'openai' package to use ChatClient.") from exc

        api_key = os.getenv(self.config.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"Missing API key environment variable: {self.config.api_key_env}")
        base_url = None
        if self.config.base_url_env:
            base_url = os.getenv(self.config.base_url_env, self.config.default_base_url or "").strip()

        client = OpenAI(api_key=api_key, base_url=base_url or None)
        response = client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
