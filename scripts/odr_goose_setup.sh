#!/usr/bin/env bash
# odr_goose_setup.sh — wire Open Dream-RSI into goose, end to end.
#
# Usage:
#   ./scripts/odr_goose_setup.sh              # check + auto-configure goose + start proxy
#   ./scripts/odr_goose_setup.sh --check      # diagnose only, change nothing
#   ./scripts/odr_goose_setup.sh --no-proxy   # skip the proxy (use odr_run_once provider='goose')
#
# Every check prints PASS/FAIL/WARN. The goose extension block is written
# automatically (with a timestamped backup); foreign config entries are
# preserved byte-for-byte.
set -u
FAILED=0
CHECK_ONLY=0
USE_PROXY=1
PY="${ODR_PY:-$(command -v python3)}"
GOOSE_DIR="$HOME/.config/goose"
GCFG="$GOOSE_DIR/config.yaml"
PROXY_PORT="${ODR_PROXY_PORT:-8799}"

for arg in "$@"; do
  case "$arg" in
    --check)    CHECK_ONLY=1 ;;
    --no-proxy) USE_PROXY=0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

say(){ printf '\n\033[1m== %s\033[0m\n' "$1"; }
ok(){ echo "  PASS: $1"; }
no(){ echo "  FAIL: $1"; FAILED=1; }
warn(){ echo "  WARN: $1"; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

say "1/7 repo + install ($REPO)"
git -C "$REPO" pull origin main >/dev/null 2>&1 && ok "updated (main)" || warn "git pull failed — continuing with the local tree"
if ! "$PY" -c "import open_dream_rsi" 2>/dev/null; then
  "$PY" -m pip install -e "$REPO" -q --user 2>/dev/null \
    || "$PY" -m pip install -e "$REPO" -q --user --break-system-packages 2>/dev/null
fi
"$PY" -c "import open_dream_rsi;print('  PASS: module', open_dream_rsi.__version__)" \
  || { no "open_dream_rsi not importable for $PY"; exit 1; }
# Prefer the same interpreter goose will spawn (macOS Homebrew python when present).
if [ -x /opt/homebrew/bin/python3 ]; then
  /opt/homebrew/bin/python3 -c "import open_dream_rsi" 2>/dev/null && PY=/opt/homebrew/bin/python3 \
    && ok "extension will use /opt/homebrew/bin/python3" \
    || warn "/opt/homebrew/bin/python3 cannot import the module — falling back to $PY (run: /opt/homebrew/bin/python3 -m pip install -e $REPO)"
fi
[ -f "$REPO/tasks.json" ] && ok "tasks.json present" || warn "no tasks.json yet — odr_add_task creates it on first use"

say "2/7 goose provider"
[ -f "$GCFG" ] && ok "config: $GCFG" || { no "missing $GCFG — set a provider in goose (Settings -> Models) first"; exit 1; }
AP=$(grep -E '^active_provider:' "$GCFG" | head -1 | cut -d: -f2- | tr -d ' ')
GP=$(grep -E '^GOOSE_PROVIDER:' "$GCFG" | head -1 | cut -d: -f2- | tr -d ' ')
PROVIDER="${GP:-$AP}"
[ -n "${PROVIDER:-}" ] && ok "provider: $PROVIDER" || no "no GOOSE_PROVIDER/active_provider found"
if ls "$GOOSE_DIR"/custom_providers/*.json >/dev/null 2>&1; then
  ok "custom_providers/*.json present:"
  for f in "$GOOSE_DIR"/custom_providers/*.json; do
    echo "    $(basename "$f") -> $(grep -o '"base_url"[^,]*' "$f" | head -1)"
  done
else
  warn "no custom_providers/*.json (fine for builtin providers; custom providers need base_url in the providers: block)"
fi
[ -f "$GOOSE_DIR/secrets.yaml" ] && ok "secrets.yaml present" || warn "no secrets.yaml — key resolution falls back to env/keychain; if step 4 shows api_key:null add --api-key support (see docs/integrations.md)"

say "3/7 upstream resolution (goose brain -> dreamer)"
RES="$("$PY" -m open_dream_rsi proxy --print-config 2>&1)"
echo "$RES" | sed 's/^/    /'
if echo "$RES" | grep -q '"base_url"'; then
  ok "upstream ok"
else
  no "upstream unresolved"
  echo "    -> add 'base_url: http://HOST:PORT/v1' under providers/$PROVIDER in $GCFG,"
  echo "       or run the proxy explicitly: $PY -m open_dream_rsi proxy --base-url http://HOST:PORT/v1 --api-key ***"
  exit 1
fi

say "4/7 credential"
if echo "$RES" | grep -q '"api_key": *null'; then
  warn "no credential found — check 'security find-generic-password -s goose' entries or export the provider *_API_KEY / OPENAI_API_KEY before starting the proxy"
else
  ok "credential resolved"
fi

say "5/7 proxy on :$PROXY_PORT"
if [ "$USE_PROXY" = 0 ]; then
  warn "skipped (--no-proxy): odr_run_once must pass provider='goose' and the MCP server needs GOOSE config readable in-session"
elif [ "$CHECK_ONLY" = 1 ]; then
  ok "skipped (--check)"
else
  pkill -f "open_dream_rsi proxy" 2>/dev/null && echo "  (stale proxy killed)"
  nohup "$PY" -m open_dream_rsi proxy --port "$PROXY_PORT" >/tmp/odr-proxy.log 2>&1 &
  sleep 1
  HEALTH=$(curl -s --max-time 5 "http://127.0.0.1:$PROXY_PORT/health" || true)
  if echo "$HEALTH" | grep -q '"status": *"ok"'; then
    ok "health ok: $(echo "$HEALTH" | head -c 200)"
  else
    no "proxy not answering — /tmp/odr-proxy.log:"; tail -5 /tmp/odr-proxy.log | sed 's/^/    /'
    exit 1
  fi
fi

say "6/7 sample completion through the proxy"
if [ "$USE_PROXY" = 0 ] || [ "$CHECK_ONLY" = 1 ]; then
  ok "skipped"
else
  CODE=$("$PY" - <<'PYEOF'
import json, sys, urllib.request
req = urllib.request.Request("http://127.0.0.1:8799/v1/chat/completions",
    data=json.dumps({"messages": [{"role": "user", "content": "Reply with exactly: OK"}],
                     "max_tokens": 64}).encode(),
    headers={"Content-Type": "application/json"}, method="POST")
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    m = d["choices"][0]["message"]
    text = m.get("content") or m.get("reasoning")
    print("PASS: completion: " + (text or "<empty>")[:80]); sys.exit(0)
except Exception as e:
    print("FAIL: " + str(e)[:200]); sys.exit(1)
PYEOF
)
  echo "  $CODE"
  echo "$CODE" | grep -q '^FAIL' && FAILED=1
fi

say "7/7 goose extension block"
CFG_ARGS=(--config "$GCFG" --repo-dir "$REPO" --py "$PY")
[ "$USE_PROXY" = 0 ] && CFG_ARGS+=(--no-proxy)
[ "$CHECK_ONLY" = 1 ] && CFG_ARGS+=(--check)
OUT="$("$PY" -m open_dream_rsi.utils.goose_config "${CFG_ARGS[@]}")"
echo "$OUT" | sed 's/^/    /'
echo "$OUT" | grep -q '"verified": true\|"action": "check-only"\|"action": "unchanged"' \
  && ok "extension block installed" || no "extension block write failed"

if [ "$FAILED" = 0 ]; then
  printf '\n\033[32mALL CHECKS PASSED\033[0m — next steps:\n'
  cat <<EOF
  1. Fully quit goose (Cmd+Q, not just a new window) and relaunch — the
     desktop app caches config.yaml at startup.
  2. New session -> extensions picker (+) -> activate 'open-dream-rsi'.
  3. First message: "Call the odr_status tool and show its raw output."
  4. The proxy must keep running while you work (or install a launchd agent:
     see docs/integrations.md). Proxy log: /tmp/odr-proxy.log
EOF
else
  printf '\n\033[31mFAILURES above — fix and re-run\033[0m\n'
fi
exit "$FAILED"
