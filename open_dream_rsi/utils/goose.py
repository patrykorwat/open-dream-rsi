"""Discovery of goose's own model configuration (provider, model, keys).

goose keeps everything under ``~/.config/goose/``:

* ``config.yaml``          — ``GOOSE_PROVIDER`` / ``GOOSE_MODEL`` / ``OPENAI_HOST`` ...
* ``secrets.yaml``         — fallback key storage (used when the keyring is disabled)
* ``custom_providers/*.json`` — declarative OpenAI/Anthropic/Ollama-compatible providers
* the OS keychain          — the real keys (probed on macOS via ``security``)

Open Dream-RSI reads these files so the dreamer reuses **exactly the same
brain as the goose session** — one provider, one model, one credential, no
duplicated keys. Used by :mod:`open_dream_rsi.proxy` and by the ``goose``
LLM provider (``odr_run_once`` / ``--provider goose``).

Resolution precedence: explicit call-time overrides > goose's own files
(config.yaml / custom_providers / secrets.yaml) > process environment >
built-in defaults. Standard library only.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

#: Env var that overrides the goose config dir (useful in tests / CI).
CONFIG_DIR_ENV = "ODR_GOOSE_CONFIG_DIR"

DEFAULT_OPENAI_HOST = "https://api.openai.com"
DEFAULT_OPENAI_BASE_PATH = "v1/chat/completions"
DEFAULT_ANTHROPIC_HOST = "https://api.anthropic.com"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"

KNOWN_ENGINES = ("openai", "anthropic", "ollama")

_KEY_ENV_BY_ENGINE = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": "OLLAMA_API_KEY",
}


class GooseConfigError(RuntimeError):
    """Raised when goose's configuration cannot be resolved into an upstream."""


@dataclass
class ResolvedUpstream:
    """Everything needed to talk to the model goose itself is using."""

    provider: str
    engine: str  # openai | anthropic | ollama
    model: Optional[str]
    base_url: str  # OpenAI-style base for openai/ollama; host for anthropic
    api_key: Optional[str] = None
    api_key_source: Optional[str] = None  # where the key was found (diagnostics)
    headers: Dict[str, str] = field(default_factory=dict)
    config_dir: str = ""

    def masked(self) -> Dict[str, Any]:
        """JSON-safe summary with the credential masked (for /health, CLI)."""
        key = self.api_key
        shown: Optional[str] = None
        if key:
            shown = (key[:2] + "***") if len(key) > 6 else "***"
        return {
            "provider": self.provider,
            "engine": self.engine,
            "model": self.model,
            "base_url": self.base_url,
            "api_key": shown,
            "api_key_source": self.api_key_source,
            "config_dir": self.config_dir,
        }


# ---------------------------------------------------------------------------
# Minimal flat-YAML reader (stdlib-only by design)
# ---------------------------------------------------------------------------


def _flatten_yaml(path: Path) -> Dict[str, str]:
    """Parse top-level ``key: value`` scalars from a small YAML file.

    Nested blocks, lists and comments are ignored — goose's config.yaml and
    secrets.yaml keep every value we need at the top level.
    """
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#") or raw[0] in " \t-":
            continue
        key, sep, value = raw.partition(":")
        if not sep:
            continue
        value = value.split(" #", 1)[0].strip().strip("'\"")
        if value:
            out[key.strip()] = value
    return out


def _nested_block(path: Path, block: str) -> Dict[str, Dict[str, str]]:
    """Parse a two-level ``block:`` mapping from goose's config.yaml.

    The goose desktop app writes e.g.::

        providers:
          custom_spark-27b7:
            enabled: true
            model: local-inference-lab/Qwen3.8-Flash-Next-NVFP4

    which the flat reader skips (indented lines). Returns
    ``{name: {key: value}}``; empty when the block is absent.
    """
    out: Dict[str, Dict[str, str]] = {}
    if not path.exists():
        return out
    in_block = False
    current: Optional[Dict[str, str]] = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        if indent == 0:
            in_block = raw.strip().rstrip(":") == block
            current = None
            continue
        if not in_block:
            continue
        stripped = raw.strip()
        key, sep, value = stripped.partition(":")
        if not sep:
            continue
        key, value = key.strip(), value.split(" #", 1)[0].strip().strip("'\"")
        if indent <= 2 and not value:          # provider name line
            current = {}
            out[key] = current
        elif current is not None and value:    # property line
            current[key] = value
    return out


def goose_config_dir(override: Optional[str] = None) -> Path:
    """Locate the goose config directory (``~/.config/goose`` by default)."""
    root = override or os.environ.get(CONFIG_DIR_ENV)
    if root:
        return Path(root).expanduser()
    return Path.home() / ".config" / "goose"


def _derive_openai_base(url: str) -> str:
    """Normalise a URL to an OpenAI-compatible *base* (``.../v1`` style)."""
    url = url.rstrip("/")
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")]
    return url


def _compose_openai_base(host: str, base_path: str) -> str:
    """Compose a base URL from goose's ``OPENAI_HOST`` + ``OPENAI_BASE_PATH``."""
    host = host.rstrip("/")
    bp = "/" + (base_path or "").strip("/")
    if bp.endswith("/chat/completions"):
        bp = bp[: -len("/chat/completions")]
    if bp in ("", "/"):
        return host if host.endswith("/v1") else host + "/v1"
    return host + bp


def _is_local(base_url: str) -> bool:
    return any(h in base_url for h in ("127.0.0.1", "://localhost", "//[::1]"))


# ---------------------------------------------------------------------------
# Key discovery (files > env > macOS keychain)
# ---------------------------------------------------------------------------


def _keychain(secret_name: str) -> Optional[str]:
    """Best-effort macOS keychain probe under the ``goose`` service name."""
    if sys.platform != "darwin" or not shutil.which("security"):
        return None
    candidates = [secret_name, f"GOOSE_{secret_name}", f"goose_{secret_name}"]
    for account in candidates:
        try:
            proc = subprocess.run(
                ["security", "find-generic-password", "-s", "goose", "-a", account, "-w"],
                capture_output=True, text=True, timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    return None


def _find_key(secret_name: str, explicit: Optional[str],
              secrets: Dict[str, str], env: Any) -> tuple:
    if explicit:
        return explicit, "override"
    names = (secret_name, f"GOOSE_{secret_name}")
    for cand in names:
        if secrets.get(cand):
            return secrets[cand], f"secrets.yaml:{cand}"
    for cand in names:
        if env.get(cand):
            return env[cand], f"environment:{cand}"
    kc = _keychain(secret_name)
    if kc:
        return kc, "macOS keychain"
    return None, None


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


def _find_custom_provider(cdir: Path, provider: str) -> Optional[Dict[str, Any]]:
    cdir = cdir / "custom_providers"
    if not cdir.is_dir():
        return None
    for path in sorted(cdir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if provider in (data.get("name"), data.get("display_name")):
            return data
    return None


def resolve_goose(
    config_dir: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ResolvedUpstream:
    """Resolve goose's provider/model/credential into a usable upstream.

    Raises :class:`GooseConfigError` when nothing can be resolved. Missing
    keys are *not* an error here (the caller decides; local endpoints get a
    dummy bearer token) — the credential requirement is enforced at POST time.
    """
    cdir = goose_config_dir(config_dir)
    settings = _flatten_yaml(cdir / "config.yaml")
    secrets = _flatten_yaml(cdir / "secrets.yaml")
    env = os.environ
    providers_block = _nested_block(cdir / "config.yaml", "providers")

    provider = (provider or env.get("GOOSE_PROVIDER") or settings.get("GOOSE_PROVIDER")
                # goose desktop app format: active_provider + nested providers: block
                or settings.get("active_provider"))
    if not provider:
        raise GooseConfigError(
            f"GOOSE_PROVIDER / active_provider not found in {cdir / 'config.yaml'} — "
            "pass --provider explicitly"
        )
    model = (model or env.get("GOOSE_MODEL") or settings.get("GOOSE_MODEL")
             or (providers_block.get(provider) or {}).get("model"))

    custom = _find_custom_provider(cdir, provider)
    engine = (custom or {}).get("engine") or (provider if provider in KNOWN_ENGINES else None)
    # Desktop-app custom providers ("custom_*") usually live in
    # custom_providers/*.json, but may be declared only by the nested
    # providers: block — such names speak the OpenAI protocol.
    if engine is None and (provider.startswith("custom_")
                           or provider in providers_block):
        engine = "openai"
    if engine not in KNOWN_ENGINES:
        raise GooseConfigError(
            f"provider '{provider}' is neither openai/anthropic/ollama nor a "
            f"custom_providers entry — point --base-url at an OpenAI-compatible endpoint"
        )

    headers = {str(k): str(v) for k, v in ((custom or {}).get("headers") or {}).items()}

    block = providers_block.get(provider) or {}
    resolved_base: str
    block_url = next((block[k] for k in ("base_url", "api_url", "url", "host", "api_base")
                      if block.get(k)), None)
    if base_url:
        resolved_base = _derive_openai_base(base_url) if engine == "openai" else base_url.rstrip("/")
    elif custom and custom.get("base_url"):
        raw = custom["base_url"]
        resolved_base = _derive_openai_base(raw) if engine == "openai" else raw.rstrip("/")
    elif block_url:
        raw = block_url
        resolved_base = _derive_openai_base(raw) if engine == "openai" else raw.rstrip("/")
    elif engine == "openai":
        up_name = provider.replace("-", "_").replace(".", "_").upper()
        host = (settings.get("OPENAI_HOST") or settings.get("OPENAI_BASE_URL")
                or env.get("OPENAI_HOST") or env.get("OPENAI_BASE_URL")
                or env.get(f"{up_name}_HOST") or env.get(f"{up_name}_API_URL")
                or (DEFAULT_OPENAI_HOST if provider in KNOWN_ENGINES else None))
        if not host:
            raise GooseConfigError(
                f"custom provider '{provider}' has no base_url in "
                f"{cdir / 'custom_providers'} or the providers: block — pass "
                "--base-url (or export " + f"{up_name}_HOST)")
        base_path = (settings.get("OPENAI_BASE_PATH") or env.get("OPENAI_BASE_PATH")
                     or DEFAULT_OPENAI_BASE_PATH)
        resolved_base = _compose_openai_base(host, base_path)
    elif engine == "anthropic":
        host = settings.get("ANTHROPIC_HOST") or env.get("ANTHROPIC_HOST") or DEFAULT_ANTHROPIC_HOST
        resolved_base = host.rstrip("/")
    else:  # ollama — speaks the OpenAI protocol under /v1
        host = settings.get("OLLAMA_HOST") or env.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST
        if "://" not in host:
            host = "http://" + host
        resolved_base = host.rstrip("/")
        if not resolved_base.endswith("/v1"):
            resolved_base += "/v1"

    key_env = (custom or {}).get("api_key_env")
    if not key_env and (provider.startswith("custom_") or provider in providers_block):
        # goose desktop app convention for custom providers: <NAME>_API_KEY
        key_env = f"{provider.replace('-', '_').replace('.', '_').upper()}_API_KEY"
    key, source = _find_key(key_env or _KEY_ENV_BY_ENGINE[engine], api_key, secrets, env)
    if key is None and (custom or provider in providers_block or provider.startswith("custom_")):
        # try the plain engine key as last resort (shared credentials)
        key, source = _find_key(_KEY_ENV_BY_ENGINE[engine], api_key, secrets, env)
    if key is None and engine != "anthropic" and _is_local(resolved_base):
        key, source = "local", "local-endpoint"  # vLLM/Ollama accept any bearer

    return ResolvedUpstream(
        provider=provider, engine=engine, model=model, base_url=resolved_base,
        api_key=key, api_key_source=source, headers=headers, config_dir=str(cdir),
    )


def goose_available(config_dir: Optional[str] = None) -> bool:
    """True when a goose config directory with a provider exists."""
    try:
        resolve_goose(config_dir)
        return True
    except GooseConfigError:
        return False
