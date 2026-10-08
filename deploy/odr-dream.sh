#!/usr/bin/env bash
# =============================================================================
# odr-dream — automated Dream-RSI cycle for the Hermes container (issue #3)
#
# The ALM (Artifact Lifecycle Manager) is deterministic machinery wired into
# the loop's seams; it only moves when the loop runs. This cron driver makes
# the whole pipeline self-running:
#
#   1. evidence   — export new Hermes session transcripts (redacted)
#   2. candidates — distil failure-shaped episodes into STAGING lessons
#   3. learning   — one improvement cycle: online attempts + dreaming +
#                   paired-replay promotion gate + ALM maintenance
#                   (stale detection, supersession, archive, GC)
#   4. publish    — render ACTIVE lessons into discovered Hermes skills
#
# Nothing here activates a lesson by hand: promotion is the gate's decision,
# lifecycle transitions are the ALM's, recorded as immutable events.
# =============================================================================
set -uo pipefail

MEM=/opt/data/dream_rsi
EP=$MEM/episodes.jsonl
SKILLS=/opt/data/skills/odr-curated
LOG=/home/porwat/docker/hermes/odr-dream.log
ENVV='PYTHONPATH=/opt/odr OPENAI_BASE_URL=http://192.168.0.12:8000/v1 ODR_LLM_MODEL=azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound ODR_LLM_NO_THINKING=1'

run() { docker exec hermes bash -c "cd /opt/odr && $ENVV $*"; }

{
  echo "=== odr-dream $(date -Is) ==="

  # seed the task queue so 'loop' has something to chew on even on first run
  docker exec hermes bash -c "mkdir -p $MEM && [ -f $MEM/tasks.json ] || echo '[]' > $MEM/tasks.json"

  # 1. evidence: transcripts from this Hermes instance (read-only, redacted)
  run "python3 -m open_dream_rsi --memory $MEM sessions export \
      --source hermes --db /opt/data/state.db --limit 20 --out $EP" \
      && echo "[ok] sessions export" || echo "[warn] sessions export failed"

  # 2. candidates: staging lessons from failure-shaped episodes
  run "python3 -m open_dream_rsi --memory $MEM sessions distill $EP --limit 20 \
      --provider local" \
      && echo "[ok] distill" || echo "[warn] distill failed"

  # 3. learning + lifecycle: promotion gate + ALM maintenance
  run "python3 -m open_dream_rsi --memory $MEM loop --tasks $MEM/tasks.json \
      --provider local --budget 20 --once" \
      && echo "[ok] loop cycle" || echo "[warn] loop cycle failed"

  # 4. publish: earned lessons -> SKILL.md files Hermes discovers
  run "python3 -m open_dream_rsi --memory $MEM skills export --out $SKILLS" \
      && echo "[ok] skills export" || echo "[warn] skills export failed"

  echo "=== done $(date -Is) ==="
} >> "$LOG" 2>&1
