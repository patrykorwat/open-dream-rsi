#!/usr/bin/env bash
# odr_fleet_dream.sh — run one `odr dream` cycle per workspace memory dir.
#
# The timer half of a fleet deployment: a systemd oneshot unit (or cron job)
# calls THIS script; it loops every directory matching a glob and fires one
# dream cycle per memory dir with the LLM env merged from an env file.
# A typical fleet unit is documented in README "Fleet deployments".
#
# Configuration (environment; all optional except ODR_WS_GLOB in practice):
#   ODR_PY            python interpreter        (default: python3)
#   ODR_PKG           dir containing open_dream_rsi/ -> PYTHONPATH
#                     (e.g. /opt/odr/share/open-dream-rsi; default: unset,
#                      the module must then already be importable)
#   ODR_WS_GLOB       glob of MEMORY DIRS, one dream per match
#                     (default: ~/.dream_rsi; fleet example:
#                      /opt/odr/workspaces/*/odr-mem — no spaces)
#   ODR_SOURCE        cursor | hermes | claude   (default: cursor)
#   ODR_PROVIDER      cursor-agent (default) | openai | local | goose
#   ODR_MODEL         optional model pin (prefer unset: auto-discovery)
#   ODR_ENV_FILE      KEY=VALUE file sourced into the dream env
#                     (OPENAI_BASE_URL / OPENAI_API_KEY / ...)
#   ODR_CURSOR_USER_DIR  optional --cursor-user-dir override
#   ODR_SESSIONS_DB   required when ODR_SOURCE=hermes (the state.db path)
#   ODR_INGEST        1 (default) runs the evidence ingest; 0 skips --source
#                     (heartbeat-only cycles)
#   ODR_BUDGET        --budget per full dream   (default: 100)
#   ODR_STAGGER       seconds between workspaces (default: 1)
#
# One cycle per call — wire the script into a systemd timer (the ExecStart=
# of a Type=oneshot unit, see README "Fleet deployments") or a cron line;
# the dream module itself decides maintenance vs full per workspace (and the
# wake self-test only fires on a FULL dream with an empty queue).
set -u

PY="${ODR_PY:-python3}"
PKG="${ODR_PKG:-}"
GLOB="${ODR_WS_GLOB:-$HOME/.dream_rsi}"
SOURCE="${ODR_SOURCE:-cursor}"
PROVIDER="${ODR_PROVIDER:-cursor-agent}"
MODEL="${ODR_MODEL:-}"
ENV_FILE="${ODR_ENV_FILE:-}"
INGEST="${ODR_INGEST:-1}"
STAGGER="${ODR_STAGGER:-1}"
BUDGET="${ODR_BUDGET:-100}"

say(){ printf '%s\n' "$*"; }

if [ -n "$ENV_FILE" ]; then
  if [ -f "$ENV_FILE" ]; then
    set -a; . "$ENV_FILE"; set +a
  else
    say "FAIL: ODR_ENV_FILE not found: $ENV_FILE"
    exit 2
  fi
fi

if [ -n "$PKG" ]; then
  PYTHONPATH="$PKG${PYTHONPATH:+:$PYTHONPATH}"
  export PYTHONPATH
fi

if ! "$PY" -c "import open_dream_rsi" >/dev/null 2>&1; then
  say "FAIL: open_dream_rsi not importable by $PY (set ODR_PKG to the dir containing open_dream_rsi/)"
  exit 2
fi

if [ "$SOURCE" = "hermes" ] && [ -z "${ODR_SESSIONS_DB:-}" ]; then
  say "FAIL: ODR_SOURCE=hermes needs ODR_SESSION_DB (the Hermes state.db path)"
  exit 2
fi

# shellcheck disable=SC2206  # glob expansion is the point
dirs=( $GLOB )
if [ "${#dirs[@]}" -eq 0 ] || [ ! -e "${dirs[0]:-}" ]; then
  say "FAIL: no memory dirs match ODR_WS_GLOB: $GLOB"
  exit 1
fi

ok=0; fail=0
for d in "${dirs[@]}"; do
  args=( -m open_dream_rsi --memory "$d" dream --provider "$PROVIDER" --budget "$BUDGET" )
  if [ "$INGEST" = "1" ]; then
    args+=( --source "$SOURCE" )
    [ -n "${ODR_SESSIONS_DB:-}" ] && args+=( --sessions "$ODR_SESSIONS_DB" )
    [ -n "${ODR_CURSOR_USER_DIR:-}" ] && args+=( --cursor-user-dir "$ODR_CURSOR_USER_DIR" )
  fi
  [ -n "$MODEL" ] && args+=( --model "$MODEL" )

  say "--- $d ($(date -u +%FT%TZ))"
  if "$PY" "${args[@]}"; then
    ok=$((ok+1))
  else
    fail=$((fail+1))
    say "FAIL: dream rc=$? on $d"
  fi
  [ "$STAGGER" -gt 0 ] && sleep "$STAGGER"
done
say "fleet dream done: ok=$ok fail=$fail"
[ "$fail" -eq 0 ]
