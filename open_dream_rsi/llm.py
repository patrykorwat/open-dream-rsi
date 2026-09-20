"""LLM integration layer via an OpenAI-compatible interface.

Open Dream-RSI does not require a specific provider — any endpoint that
implements the OpenAI protocol (``POST {base_url}/chat/completions``) works:

* OpenAI                            -> https://api.openai.com/v1
* Cursor Models API                 -> https://api2.cursor.sh
* local vLLM / Ollama / LM Studio   -> http://127.0.0.1:8000/v1 etc.

The client uses only the standard library (urllib), so the core package
remains free of external dependencies.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_TIMEOUT = 120.0

#: Predefined OpenAI-compatible endpoint profiles.
ENDPOINT_PRESETS: Dict[str, Dict[str, str]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
    "cursor": {
        "base_url": "https://api2.cursor.sh",
        "api_key_env": "CURSOR_API_KEY",
        "default_model": "cursor-grok-4.5-high",
    },
    "local": {
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "local-model",
    },
}


@dataclass
class LLMConfig:
    """LLM client configuration (via environment variables or explicit values)."""

    base_url: str = "https://api.openai.com/v1"
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    api_key_env: Optional[str] = None
    extra_headers: Dict[str, str] = field(default_factory=dict)
    timeout: float = DEFAULT_TIMEOUT

    @classmethod
    def from_preset(cls, preset: str, **overrides: Any) -> "LLMConfig":
        """Build a config from a predefined profile ('openai', 'cursor', 'local').

        The API key is read from the environment variable named by the profile
        (or overridden in ``overrides``). It is never stored in code.

        Precedence: explicit ``overrides`` > environment variables
        (``OPENAI_BASE_URL``, ``ODR_LLM_MODEL``) > preset defaults.
        """
        if preset not in ENDPOINT_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. Available: {sorted(ENDPOINT_PRESETS)}"
            )
        p = ENDPOINT_PRESETS[preset]
        api_key_env = overrides.pop("api_key_env", p["api_key_env"])
        base_url = overrides.pop(
            "base_url", os.environ.get("OPENAI_BASE_URL") or p["base_url"]
        )
        model = overrides.pop("model", os.environ.get("ODR_LLM_MODEL") or p["default_model"])
        cfg = cls(
            base_url=base_url,
            model=model,
            api_key_env=api_key_env,
            **overrides,
        )
        if cfg.api_key is None and api_key_env:
            cfg.api_key = os.environ.get(api_key_env)
        return cfg

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Configure from the environment: OPENAI_BASE_URL / OPENAI_API_KEY / ODR_LLM_MODEL.

        ``ODR_LLM_PRESET=cursor`` switches to the Cursor Models API profile.
        """
        preset = os.environ.get("ODR_LLM_PRESET", "").lower()
        if preset:
            overrides: Dict[str, Any] = {}
            if os.environ.get("OPENAI_BASE_URL"):
                overrides["base_url"] = os.environ["OPENAI_BASE_URL"]
            if os.environ.get("ODR_LLM_MODEL"):
                overrides["model"] = os.environ["ODR_LLM_MODEL"]
            return cls.from_preset(preset, **overrides)
        return cls(
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=os.environ.get("ODR_LLM_MODEL", "gpt-4o-mini"),
        )


class LLMError(RuntimeError):
    """Error raised when an LLM endpoint call fails."""


class OpenAICompatibleClient:
    """Minimal chat-completions client compatible with the OpenAI protocol.

    Works with OpenAI, the Cursor Models API and any local server
    (vLLM, Ollama, LM Studio) exposing ``/chat/completions``.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()

    # -- public API -------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Send chat messages and return the assistant's reply text."""
        payload = {
            "model": model or self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = self._post("/chat/completions", payload)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:  # unexpected response shape
            raise LLMError(f"Unexpected API response: {data!r}") from exc

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """Convenience single-prompt call: system (optional) + user."""
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, temperature=temperature)

    # -- transport ----------------------------------------------------------------

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.config.api_key:
            raise LLMError(
                "Missing API key. Set the "
                f"{self.config.api_key_env or 'OPENAI_API_KEY'} environment variable "
                "or pass LLMConfig(api_key=...)."
            )
        url = self.config.base_url.rstrip("/") + path
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
            **self.config.extra_headers,
        }
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            raise LLMError(f"HTTP {exc.code} from {url}: {body}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"Could not connect to {url}: {exc.reason}") from exc


class StubClient:
    """Deterministic fallback client — lets you test the loop without an LLM."""

    def __init__(self, response: str = "explore"):
        self.response = response
        self.calls: List[List[Dict[str, str]]] = []

    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None,
             temperature: float = 0.7, max_tokens: int = 1024) -> str:
        self.calls.append(messages)
        return self.response

    def complete(self, prompt: str, system: Optional[str] = None, **_: Any) -> str:
        return self.chat([{"role": "user", "content": prompt}])
