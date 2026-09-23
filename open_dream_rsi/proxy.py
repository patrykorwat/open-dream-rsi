"""OpenAI-compatible proxy that forwards to the model *goose* itself uses.

Why: goose spawns MCP extensions with a scrubbed environment and no LLM
credential, and (as of the 2025-xx protocol line) does not implement MCP
sampling — so an extension cannot borrow the host's model through the
protocol. This proxy closes that gap out-of-band: it reads goose's own
configuration (:mod:`open_dream_rsi.utils.goose`) and exposes a plain
``POST /v1/chat/completions`` on localhost that forwards to the very same
provider, model and credential goose uses. One brain, one key, zero key
duplication in extension configs.

Run::

    python -m open_dream_rsi proxy [--port 8799] [--provider ...] [--model ...]

Then point anything OpenAI-compatible at it, e.g. in goose's own
``config.yaml`` extension block::

    envs:
      OPENAI_BASE_URL: "http://127.0.0.1:8799/v1"
      OPENAI_API_KEY: "pr..."

Or use the direct path without a daemon at all: ``--provider goose`` in
``loop`` / ``odr_run_once`` resolves goose's config at call time.

Endpoints:

* ``POST /v1/chat/completions`` — forward (OpenAI upstream: byte passthrough
  incl. SSE streaming; Anthropic upstream: minimal messages translation).
* ``POST /v1/completions``      — accepted (prompt mapped to one user message).
* ``GET  /v1/models``           — the resolved model id.
* ``GET  /health``              — resolved upstream with the key masked.

Stdlib only, bound to 127.0.0.1 by default — like every other component.
"""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple

from open_dream_rsi.utils.goose import GooseConfigError, ResolvedUpstream, resolve_goose

DEFAULT_PORT = 8799
DEFAULT_TIMEOUT = 300.0


class UpstreamError(RuntimeError):
    """Raised when forwarding to the resolved upstream fails."""


# ---------------------------------------------------------------------------
# Upstream calls (OpenAI + Anthropic translation)
# ---------------------------------------------------------------------------


def _post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str],
               timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise UpstreamError(f"HTTP {exc.code} from {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise UpstreamError(f"Could not connect to {url}: {exc.reason}") from exc


def _openai_headers(up: ResolvedUpstream) -> Dict[str, str]:
    return {"Authorization": f"Bearer {up.api_key or 'local'}", **up.headers}


def forward_chat_completions(up: ResolvedUpstream, body: Dict[str, Any],
                             timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """Forward an OpenAI chat-completions request to the resolved upstream."""
    if not up.api_key:
        raise UpstreamError(
            f"no credential for provider '{up.provider}' — store it in goose "
            "(keychain or ~/.config/goose/secrets.yaml) or pass --api-key")
    model = body.get("model") or up.model
    if not model:
        raise UpstreamError("request has no 'model' and goose config has no GOOSE_MODEL")
    payload = dict(body, model=model)

    if up.engine == "anthropic":
        return _forward_anthropic(up, payload, timeout)

    if payload.pop("stream", False):
        return _forward_openai_stream(up, payload, timeout)

    return _post_json(up.base_url.rstrip("/") + "/chat/completions",
                      payload, _openai_headers(up), timeout)


def _forward_openai_stream(up: ResolvedUpstream, payload: Dict[str, Any],
                           timeout: float) -> Dict[str, Any]:
    """Collect an SSE stream into a single completion dict.

    The proxy is consumed by the ODR client (non-streaming). Buffering the
    stream keeps the implementation single-path and stdlib-only while still
    accepting ``stream: true`` from naive callers.
    """
    url = up.base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **_openai_headers(up)},
        method="POST")
    text_parts = []
    final: Optional[Dict[str, Any]] = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("choices"):
                    delta = chunk["choices"][0].get("delta") or {}
                    if delta.get("content"):
                        text_parts.append(delta["content"])
                    if chunk["choices"][0].get("finish_reason"):
                        final = chunk
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise UpstreamError(f"HTTP {exc.code} from {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise UpstreamError(f"Could not connect to {url}: {exc.reason}") from exc
    base = final or {"model": payload["model"], "choices": [{"index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop"}]}
    base.setdefault("choices", [{"index": 0, "message": {}}])
    base["choices"][0].setdefault("message", {})["content"] = "".join(text_parts)
    base["choices"][0]["message"].setdefault("role", "assistant")
    base.setdefault("proxy_stream", "buffered")
    return base


def _forward_anthropic(up: ResolvedUpstream, payload: Dict[str, Any],
                       timeout: float) -> Dict[str, Any]:
    """Minimal OpenAI -> Anthropic messages translation (non-streaming)."""
    system_parts, messages = [], []
    for msg in payload.get("messages", []):
        role = msg.get("role")
        content = msg.get("content") or ""
        if role == "system":
            system_parts.append(content)
        elif role in ("user", "assistant"):
            messages.append({"role": role, "content": content})
    body = {
        "model": payload["model"],
        "messages": messages or [{"role": "user", "content": ""}],
        "max_tokens": int(payload.get("max_tokens") or 1024),
    }
    if system_parts:
        body["system"] = "\n".join(system_parts)
    if payload.get("temperature") is not None:
        body["temperature"] = payload["temperature"]
    headers = {"x-api-key": up.api_key or "",
               "anthropic-version": "2023-06-01", **up.headers}
    data = _post_json(up.base_url.rstrip("/") + "/v1/messages", body, headers, timeout)
    text = "".join(b.get("text", "") for b in data.get("content", [])
                   if b.get("type") == "text")
    return {
        "id": data.get("id", "proxy-completion"),
        "object": "chat.completion",
        "model": data.get("model", payload["model"]),
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": text},
                     "finish_reason": {"end_turn": "stop", "max_tokens": "length"}
                     .get(data.get("stop_reason"), "stop")}],
        "usage": {k: v for k, v in (data.get("usage") or {}).items()
                  if k in ("input_tokens", "output_tokens")},
    }


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------


class ProxyState:
    """Live upstream resolution + counters (re-read per request so goose can
    switch models/providers without restarting the proxy)."""

    def __init__(self, overrides: Optional[Dict[str, Any]] = None,
                 passthrough_models: bool = False):
        self.overrides = overrides or {}
        self.passthrough_models = passthrough_models
        self.requests = 0
        self.errors = 0
        self._lock = threading.Lock()

    def resolve(self) -> ResolvedUpstream:
        return resolve_goose(
            provider=self.overrides.get("provider"),
            model=self.overrides.get("model"),
            base_url=self.overrides.get("base_url"),
            api_key=self.overrides.get("api_key"),
        )

    def record_ok(self) -> None:
        with self._lock:
            self.requests += 1

    def record_err(self) -> None:
        with self._lock:
            self.errors += 1


def make_handler(state: ProxyState):
    class ProxyHandler(BaseHTTPRequestHandler):
        # HTTP/1.0 close-after-response semantics: safe for the buffered JSON
        # the proxy emits and lets naive SSE clients terminate cleanly.
        protocol_version = "HTTP/1.0"

        # -- plumbing ---------------------------------------------------------
        def _send(self, code: int, payload: Dict[str, Any]) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _read_body(self) -> Dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if not length:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def log_message(self, fmt, *args):  # keep stderr quiet unless verbose
            pass

        # -- routes -----------------------------------------------------------
        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/")
            if path in ("/health", "/v1/health"):
                try:
                    up = state.resolve()
                    self._send(200, {"status": "ok", "upstream": up.masked(),
                                     "requests": state.requests, "errors": state.errors})
                except GooseConfigError as exc:
                    self._send(503, {"status": "unresolved", "error": str(exc)})
                return
            if path == "/v1/models":
                try:
                    up = state.resolve()
                    mid = up.model or "goose-model"
                    self._send(200, {"object": "list",
                                     "data": [{"id": mid, "object": "model",
                                               "owned_by": f"goose:{up.provider}"}]})
                except GooseConfigError as exc:
                    self._send(503, {"error": str(exc)})
                return
            self._send(404, {"error": {"message": f"unknown path {path}"}})

        def do_POST(self):
            path = self.path.split("?")[0].rstrip("/")
            try:
                body = self._read_body()
            except json.JSONDecodeError as exc:
                self._send(400, {"error": {"message": f"invalid JSON: {exc}"}})
                return
            if path == "/v1/completions":  # legacy completions -> chat mapping
                body = {"model": body.get("model"), "messages": [
                    {"role": "user", "content": body.get("prompt", "")}]}
                path = "/v1/chat/completions"
            if path != "/v1/chat/completions":
                self._send(404, {"error": {"message": f"unknown path {path}"}})
                return
            try:
                up = state.resolve()
                if not state.passthrough_models and up.model:
                    body = dict(body, model=up.model)  # pin goose's exact model
                data = forward_chat_completions(up, body)
                state.record_ok()
                self._send(200, data)
            except GooseConfigError as exc:
                state.record_err()
                self._send(503, {"error": {"message": str(exc), "type": "goose_config"}})
            except UpstreamError as exc:
                state.record_err()
                self._send(502, {"error": {"message": str(exc), "type": "upstream"}})
            except Exception as exc:  # never kill the server thread
                state.record_err()
                self._send(500, {"error": {"message": f"{type(exc).__name__}: {exc}"}})

    return ProxyHandler


def serve(host: str = "127.0.0.1", port: int = DEFAULT_PORT,
          overrides: Optional[Dict[str, Any]] = None,
          passthrough_models: bool = False) -> Tuple[ThreadingHTTPServer, ProxyState]:
    """Create (but do not block on) the proxy server. Returns (server, state)."""
    state = ProxyState(overrides=overrides, passthrough_models=passthrough_models)
    httpd = ThreadingHTTPServer((host, port), make_handler(state))
    return httpd, state


def main(argv=None, serve_forever: bool = True) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="odr-proxy",
        description="OpenAI-compatible proxy forwarding to goose's own model")
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (default loopback only)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--provider", default=None,
                        help="override goose's GOOSE_PROVIDER")
    parser.add_argument("--model", default=None,
                        help="override goose's GOOSE_MODEL")
    parser.add_argument("--base-url", default=None,
                        help="override upstream base URL (OpenAI-compatible)")
    parser.add_argument("--api-key", default=None,
                        help="override upstream credential (prefer goose secrets)")
    parser.add_argument("--passthrough-models", action="store_true",
                        help="keep caller's model name instead of pinning GOOSE_MODEL")
    parser.add_argument("--print-config", action="store_true",
                        help="show the resolved upstream (key masked) and exit")
    args = parser.parse_args(argv)

    overrides = {k: v for k, v in {
        "provider": args.provider, "model": args.model,
        "base_url": args.base_url, "api_key": args.api_key}.items() if v}

    if args.print_config:
        try:
            up = resolve_goose(**overrides)
            print(json.dumps(up.masked(), indent=2))
            return 0
        except GooseConfigError as exc:
            print(f"config error: {exc}", file=sys.stderr)
            return 2

    try:
        httpd, _state = serve(args.host, args.port, overrides,
                              passthrough_models=args.passthrough_models)
    except OSError as exc:
        print(f"cannot bind {args.host}:{args.port}: {exc}", file=sys.stderr)
        return 1
    if serve_forever:
        try:
            up = resolve_goose(**overrides)
            print(f"[odr proxy] forwarding to {up.provider}/{up.model} @ {up.base_url} "
                  f"(key: {up.api_key_source or 'MISSING'})", file=sys.stderr)
        except GooseConfigError as exc:
            print(f"[odr proxy] WARNING: {exc}", file=sys.stderr)
        print(f"[odr proxy] listening on http://{args.host}:{args.port}/v1 "
              f"— point OPENAI_BASE_URL at it", file=sys.stderr)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[odr proxy] stopped.", file=sys.stderr)
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
