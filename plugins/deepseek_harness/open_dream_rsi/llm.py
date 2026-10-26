"""LLM integration layer via an OpenAI-compatible interface.

Open Dream-RSI does not require a specific provider — any endpoint that
implements the OpenAI protocol (``POST {base_url}/chat/completions``) works:

* OpenAI                            -> https://api.openai.com/v1
* Cursor (via a local proxy over    -> http://127.0.0.1:3000/v1
  the Cursor login, e.g. cursor-openai-api)
* local vLLM / Ollama / LM Studio   -> http://127.0.0.1:8000/v1 etc.

The client uses only the standard library (urllib), so the core package
remains free of external dependencies.

Unless a model is pinned (explicitly or via ``ODR_LLM_MODEL``), the first
model the endpoint serves (``GET /v1/models``) is used — a redeployed
local server is picked up automatically, no config edits needed.
"""

from __future__ import annotations

import json
import os
import random
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: Per-call HTTP timeout. 900 s default: proposal calls legitimately generate
#: multi-kB reports/code and a shared local vLLM instance queues behind other
#: consumers — at 120 s those became the dominant failure ("timed out" events
#: burning API budget). Override with ODR_LLM_TIMEOUT (seconds).
DEFAULT_TIMEOUT = float(os.environ.get("ODR_LLM_TIMEOUT", "900.0"))

#: Model discovery GET timeout (the /models listing must never stall a run).
_DISCOVERY_TIMEOUT = 30.0

#: Transient-failure retries with exponential backoff + jitter. Retried:
#: connection errors (refused/reset/remote closed) and 429/5xx responses.
#: Generation timeouts are NOT retried by default — each timed-out attempt
#: already consumed the full call timeout, and on a saturated shared endpoint
#: blind timeout retries just multiply wall clock (ODR_LLM_RETRY_TIMEOUTS=1
#: opts in). Tune attempts/pacing with ODR_LLM_RETRIES / ODR_LLM_BACKOFF.
MAX_RETRIES = int(os.environ.get("ODR_LLM_RETRIES", "3"))
BACKOFF_BASE = float(os.environ.get("ODR_LLM_BACKOFF", "5.0"))
BACKOFF_MAX = float(os.environ.get("ODR_LLM_BACKOFF_MAX", "120.0"))
RETRY_TIMEOUTS = os.environ.get("ODR_LLM_RETRY_TIMEOUTS", "") == "1"


def _retry_delay(attempt: int) -> float:
    """Full-jitter exponential backoff: ``base * 2**attempt`` capped, 0.5x..1.5x."""
    base = min(BACKOFF_BASE * (2 ** max(0, attempt)), BACKOFF_MAX)
    return base * (0.5 + random.random())


def _default_model(base_url: str, api_key: Optional[str]) -> Optional[str]:
    """Ask an OpenAI-compatible endpoint which model to use by default.

    Returns the first id served under ``GET {base_url}/models`` or ``None``
    when the listing is missing/unreachable — callers fall back to their own
    default. Keeps local deployments working across model redeploys: a
    renamed served model is picked up automatically instead of a pinned
    (and now-404ing) name.
    """
    req = urllib.request.Request(
        base_url.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=_DISCOVERY_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (OSError, ValueError):  # URLError/HTTPError/timeout, bad JSON
        return None
    models = data.get("data") if isinstance(data, dict) else None
    if not isinstance(models, list) or not models:
        return None
    first = models[0]
    return first.get("id") if isinstance(first, dict) else None


#: Predefined OpenAI-compatible endpoint profiles.
ENDPOINT_PRESETS: Dict[str, Dict[str, str]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
    # Cursor ships no public OpenAI-compatible chat endpoint: a `crsr_` API
    # key exposes the Cloud Agents API, and api2.cursor.sh is the IDE's
    # internal (gRPC) backend — direct calls here can never work. The preset
    # therefore targets a LOCAL OpenAI-compatible proxy over the Cursor login
    # (e.g. `npx cursor-openai-api serve` on :3000) — see
    # docs/integrations.md -> Cursor.
    "cursor": {
        "base_url": "http://127.0.0.1:3000/v1",
        "api_key_env": "CURSOR_API_KEY",
        "default_model": "cursor-model",
    },
    "local": {
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "local-model",
    },
}


def _resolve_goose_preset(**overrides: Any) -> "LLMConfig":
    """Build a config from goose's own provider/model/key (~/.config/goose).

    Used when preset == 'goose': the dreamer borrows the exact model the
    goose session is using — one brain, one credential, no key duplication.
    """
    from open_dream_rsi.utils.goose import resolve_goose

    up = resolve_goose(
        provider=overrides.pop("goose_provider", None),
        model=overrides.pop("goose_model", None),
        base_url=overrides.pop("base_url", None),
        api_key=overrides.pop("api_key", None),
    )
    return LLMConfig(
        base_url=up.base_url, model=up.model or "goose-model",
        api_key=up.api_key, api_key_env=None, extra_headers=dict(up.headers),
    )


@dataclass
class LLMConfig:
    """LLM client configuration (via environment variables or explicit values)."""

    base_url: str = "https://api.openai.com/v1"
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    api_key_env: Optional[str] = None
    extra_headers: Dict[str, str] = field(default_factory=dict)
    extra_payload: Dict[str, Any] = field(default_factory=dict)
    timeout: float = DEFAULT_TIMEOUT

    @classmethod
    def from_preset(cls, preset: str, **overrides: Any) -> "LLMConfig":
        """Build a config from a predefined profile ('openai', 'cursor', 'local').

        The API key is read from the environment variable named by the profile
        (or overridden in ``overrides``). It is never stored in code.

        Precedence: explicit ``overrides`` > environment variables
        (``OPENAI_BASE_URL``, ``ODR_LLM_MODEL``) > endpoint model discovery
        (first id from ``GET /v1/models``) > preset defaults.
        """
        if preset not in ENDPOINT_PRESETS:
            if preset == "goose":  # borrow goose's own provider/model/key
                return _resolve_goose_preset(**overrides)
            raise ValueError(
                f"Unknown preset '{preset}'. Available: {sorted(ENDPOINT_PRESETS)} + 'goose'"
            )
        p = ENDPOINT_PRESETS[preset]
        api_key_env = overrides.pop("api_key_env", p["api_key_env"])
        base_url = overrides.pop(
            "base_url", os.environ.get("OPENAI_BASE_URL") or p["base_url"]
        )
        env_key = os.environ.get(api_key_env) if api_key_env else None
        model = overrides.pop(
            "model",
            os.environ.get("ODR_LLM_MODEL")
            or _default_model(base_url, env_key)
            or p["default_model"],
        )
        cfg = cls(
            base_url=base_url,
            model=model,
            api_key_env=api_key_env,
            **overrides,
        )
        if cfg.api_key is None and api_key_env:
            cfg.api_key = os.environ.get(api_key_env)
        if cfg.api_key is None and preset in ("local", "cursor"):
            cfg.api_key = "local"  # local servers and the Cursor login proxy accept any bearer token
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
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        api_key = os.environ.get("OPENAI_API_KEY")
        return cls(
            base_url=base_url,
            api_key=api_key,
            model=os.environ.get("ODR_LLM_MODEL")
            or _default_model(base_url, api_key)
            or "gpt-4o-mini",
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
        #: Cumulative token usage across calls (from the OpenAI ``usage``
        #: response block). The loop logs per-cycle deltas into ``cycle``
        #: events so metrics can report burned tokens.
        self.usage_total: Dict[str, int] = {}
        self.last_usage: Optional[Dict[str, int]] = None

    # -- public API -------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Send chat messages and return the assistant's reply text."""
        base_payload = {
            "model": model or self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        payload = {**base_payload, **self.config.extra_payload}
        try:
            data = self._post("/chat/completions", payload)
        except LLMError as exc:
            text = str(exc)
            if self.config.extra_payload and "HTTP 400" in text:
                # Self-heal: endpoints that reject the extra payload (e.g. do
                # not accept chat_template_kwargs) get one retry with the
                # plain OpenAI body, and the extra flags stay dropped.
                self.config.extra_payload.clear()
                data = self._post("/chat/completions", base_payload)
            elif "HTTP 404" in text and "does not exist" in text:
                # Self-heal: a pinned model the endpoint no longer serves
                # (renamed / deployed-away local model). Re-discover once and
                # retry; the served model stays for the rest of the session.
                served = _default_model(self.config.base_url, self.config.api_key)
                if served and served != base_payload["model"]:
                    self.config.model = served
                    base_payload["model"] = served
                    payload = {**base_payload, **self.config.extra_payload}
                    data = self._post("/chat/completions", payload)
                else:
                    raise
            else:
                raise
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError) as exc:  # unexpected response shape
            raise LLMError(f"Unexpected API response: {data!r}") from exc
        content = message.get("content")
        if content:
            return content
        # Reasoning models on vLLM (e.g. Qwen NVFP4 builds) may return content=None
        # with the text in a separate 'reasoning' field, or truncate mid-reasoning.
        reasoning = message.get("reasoning")
        if reasoning:
            return reasoning
        raise LLMError(
            f"Empty completion from {model or self.config.model} "
            f"(finish_reason={data['choices'][0].get('finish_reason')}). "
            "The model may have spent the whole token budget on reasoning — "
            "raise max_tokens or check the served model."
        )

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
        body = json.dumps(payload).encode("utf-8")
        last: Optional[LLMError] = None
        for attempt in range(MAX_RETRIES + 1):
            if attempt:
                time.sleep(_retry_delay(attempt - 1))
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except (TimeoutError, socket.timeout) as exc:
                # A generation that outlived the call timeout. Retrying burns
                # the full timeout again, so it is opt-in only.
                text = f"timed out calling {url} after {self.config.timeout:g}s"
                if not RETRY_TIMEOUTS:
                    raise LLMError(text) from exc
                last = LLMError(text)
                continue
            except urllib.error.HTTPError as exc:
                text = exc.read().decode("utf-8", errors="replace")[:500]
                if exc.code == 429 or 500 <= exc.code < 600:
                    # Rate limited / server error: retryable, back off first.
                    last = LLMError(f"HTTP {exc.code} from {url}: {text}")
                    continue
                # 4xx is a bad request or dead model — no point retrying;
                # chat()'s self-heal paths take over from here.
                raise LLMError(f"HTTP {exc.code} from {url}: {text}") from exc
            except urllib.error.URLError as exc:
                reason = exc.reason
                if isinstance(reason, (TimeoutError, socket.timeout)):
                    text = f"timed out calling {url} after {self.config.timeout:g}s"
                    if not RETRY_TIMEOUTS:
                        raise LLMError(text) from exc
                    last = LLMError(text)
                    continue
                # Connection refused / DNS / unreachable — retry with backoff:
                # the endpoint is often restarting (vLLM deploys) and comes
                # back within seconds.
                last = LLMError(f"Could not connect to {url}: {reason}")
                continue
            except OSError as exc:  # connection reset, remote closed mid-call
                last = LLMError(f"Could not connect to {url}: {exc}")
                continue
            self._record_usage(data)
            return data
        raise last or LLMError(f"Request to {url} failed")

    def _record_usage(self, data: Dict[str, Any]) -> None:
        """Accumulate token usage from the response ``usage`` block, if present."""
        usage = data.get("usage") if isinstance(data, dict) else None
        if not isinstance(usage, dict):
            return
        counts = {key: int(usage.get(key, 0) or 0)
                  for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
        self.last_usage = counts
        for key, value in counts.items():
            self.usage_total[key] = self.usage_total.get(key, 0) + value


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
