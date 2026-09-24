#!/usr/bin/env bash
# odr_goose_setup.sh — wire Open Dream-RSI to goose's own model (macOS/Linux).
#
# Usage:  ./scripts/odr_goose_setup.sh
# Every check prints PASS/FAIL/WARN; the script ends with the exact
# config.yaml block + proxy command to use.
set -u
FAILED=0
PY="${ODR_PY:-$(command -v python3)}"
GCFG="$HOME/.config/goose/config.yaml"
GOOSE_DIR="$HOME/.config/goose"

say(){ printf '\n\033[1m== %s\033[0m\n' "$1"; }
ok(){ echo "  PASS: $1"; }
no(){ echo "  FAIL: $1"; FAILED=1; }
warn(){ echo "  WARN: $1"; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

say "1/6 repo + instalacja ($REPO)"
git -C "$REPO" pull origin main >/dev/null 2>&1 && ok "zaktualizowany (main)" || warn "git pull nie udany — pracuję na tym co jest"
if ! "$PY" -c "import open_dream_rsi" 2>/dev/null; then
  "$PY" -m pip install -e "$REPO" -q --user 2>/dev/null \
    || "$PY" -m pip install -e "$REPO" -q --user --break-system-packages 2>/dev/null
fi
"$PY" -c "import open_dream_rsi;print('  PASS: moduł', open_dream_rsi.__version__)" \
  || no "open_dream_rsi nieimportowalny dla $PY"

say "2/6 provider goose"
[ -f "$GCFG" ] && ok "config: $GCFG" || no "brak $GCFG — skonfiguruj providera w goose (Settings → Models)"
AP=$(grep -E '^active_provider:' "$GCFG" 2>/dev/null | head -1 | cut -d: -f2- | tr -d ' ')
GP=$(grep -E '^GOOSE_PROVIDER:' "$GCFG" 2>/dev/null | head -1 | cut -d: -f2- | tr -d ' ')
PROVIDER="${GP:-$AP}"
[ -n "${PROVIDER:-}" ] && ok "provider: $PROVIDER" || no "provider nie rozpoznany"
if ls "$GOOSE_DIR"/custom_providers/*.json >/dev/null 2>&1; then
  ok "custom_providers/*.json obecne:"
  for f in "$GOOSE_DIR"/custom_providers/*.json; do echo "    $(basename "$f") → $(grep -o '"base_url"[^,]*' "$f" | head -1)"; done
else
  warn "brak custom_providers/*.json — jeśli blok providers: nie ma base_url, podaj go niżej"
fi
if [ -f "$GOOSE_DIR/secrets.yaml" ]; then
  ok "secrets.yaml obecny (klucze: $(grep -c ':' "$GOOSE_DIR/secrets.yaml"))"
else
  warn "brak secrets.yaml — klucz będzie szukany w keychain; jeśli nic nie znajdzie, dodaj --api-key do proxy"
fi

say "3/6 resolucja upstreamu"
RES="$("$PY" -m open_dream_rsi proxy --print-config 2>&1)"
echo "$RES" | sed 's/^/    /'
if echo "$RES" | grep -q '"base_url"'; then
  BASE_URL=$(echo "$RES" | grep '"base_url"' | head -1 | sed -E 's/.*: "(.*)".*/\1/')
  ok "upstream: $BASE_URL"
else
  no "base_url nierozwiązany"
  echo "    → dopisz do config.yaml block providers/$(echo "${PROVIDER:-provider}") klucz 'base_url: http://ADRES:PORT/v1'"
  echo "      ALBO uruchom proxy z jawne: $PY -m open_dream_rsi proxy --base-url http://ADRES:PORT/v1 --api-key ..."
  exit 1
fi

say "4/6 proxy na porcie 8799"
pkill -f "open_dream_rsi proxy" 2>/dev/null && echo "  (stary proxy zabity)"
nohup "$PY" -m open_dream_rsi proxy --port 8799 >/tmp/odr-proxy.log 2>&1 &
sleep 1
HEALTH=$(curl -s --max-time 5 http://127.0.0.1:8799/health || true)
if echo "$HEALTH" | grep -q '"status": *"ok"'; then
  ok "health ok: $(echo "$HEALTH" | head -c 200)"
else
  no "proxy nie odpowiada — /tmp/odr-proxy.log:"; tail -5 /tmp/odr-proxy.log | sed 's/^/    /'
  exit 1
fi

say "5/6 przykladowe completion przez proxy"
TASKS="$REPO/tasks.json"; MEM="$REPO/.dream_rsi"
CODE=$("$PY" - "$REPO" <<'PYEOF'
import json, sys, urllib.request
base = "http://127.0.0.1:8799/v1"
req = urllib.request.Request(base + "/chat/completions",
    data=json.dumps({"messages":[{"role":"user","content":"Say OK."}],
                     "max_tokens": 64}).encode(),
    headers={"Content-Type":"application/json"}, method="POST")
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read())
    c = d["choices"][0]["message"].get("content") or d["choices"][0]["message"].get("reasoning")
    print("PASS: completion: " + (c or "<pusty>")[:80]); sys.exit(0)
except Exception as e:
    print("FAIL: " + str(e)[:200]); sys.exit(1)
PYEOF
)
echo "  $CODE"
echo "$CODE" | grep -q '^FAIL' && FAILED=1

say "6/6 co wkleic do $GCFG"
cat <<EOF

  extensions:
    open-dream-rsi:
      enabled: true
      type: stdio
      name: open-dream-rsi
      description: "Dream-RSI self-improvement loop. Use odr_add_task to queue a
        Python task with tests, odr_run_once to run an improvement cycle,
        odr_recipes/odr_lessons to reuse verified solutions, odr_status to inspect."
      cmd: $PY
      args: ["-m", "open_dream_rsi", "mcp",
             "--tasks", "$TASKS",
             "--memory", "$MEM"]
      envs:
        OPENAI_BASE_URL: "http://127.0.0.1:8799/v1"
        OPENAI_API_KEY: "pr..."            # proxy sam sie autentykuje u gory
      timeout: 300

  Potem: quit goose (pelny restart apki), w sesji jako PIERWSZA wiadomosc:
     "Call the odr_status tool and show its raw output."
  Proxy musi dzelic w trakcie pracy (ten terminal albo LaunchAgent):
     nohup $PY -m open_dream_rsi proxy --port 8799 >/tmp/odr-proxy.log 2>&1 &

EOF
[ "$FAILED" = 0 ] && echo -e "\033[32mWSZYSTKO OK\033[0m" || echo -e "\033[31mSA FAILS — napraw powyzej i powtorz\033[0m"
exit "$FAILED"
