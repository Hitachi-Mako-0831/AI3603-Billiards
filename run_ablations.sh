#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck source=/dev/null
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate billiards_rl
fi

OPPONENT="${OPPONENT:-basic}"
N_GAMES="${N_GAMES:-120}"
REPEATS="${REPEATS:-3}"
SEED_BASE="${SEED_BASE:-42}"
FIXED_SEED="${FIXED_SEED:-1}"
LOG_DIR="${LOG_DIR:-logs/ablations/${OPPONENT}}"

mkdir -p "$LOG_DIR"

run_variant() {
  local name="$1"
  shift
  local extra_args=("$@")

  for r in $(seq 0 $((REPEATS - 1))); do
    local seed=$((SEED_BASE + r))
    local log_path="${LOG_DIR}/${name}_seed${seed}.jsonl"
    rm -f "$log_path"

    local args=(--opponent "$OPPONENT" --n_games "$N_GAMES" --log_path "$log_path")
    if [[ "$FIXED_SEED" == "1" ]]; then
      args+=(--fixed_seed --seed "$seed")
    fi
    python evaluate.py "${args[@]}" "${extra_args[@]}"
    python analyze_eval_log.py --log "$log_path" --our PhysicsAgent --limit_illegal_print 0
  done
}

run_variant full
run_variant no_safety_gate --no_safety_gate
run_variant no_robust_rollout --no_robust_rollout
run_variant no_risk_shaping --no_risk_shaping
run_variant no_fallback --no_fallback

python - <<'PY'
import glob
import json
import os
from collections import defaultdict

log_dir = os.environ.get("LOG_DIR")
if not log_dir:
    opponent = os.environ.get("OPPONENT", "basic")
    log_dir = os.path.join("logs", "ablations", opponent)

def summarize(path: str):
    W = L = D = 0
    b8_illegal = 0
    b8_cue8 = 0
    for line in open(path, "r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("type") != "game_end":
            continue
        winner = rec.get("winner_agent_class")
        reason = rec.get("winner_reason")
        if winner == "PhysicsAgent":
            W += 1
        elif winner == "SAME":
            D += 1
        else:
            L += 1
            if reason == "ILLEGAL_8_POCKETED":
                b8_illegal += 1
            elif reason == "CUE_AND_8_FOUL_LOSS":
                b8_cue8 += 1
    n = W + L + D
    rate = (W + 0.5 * D) / n if n else 0.0
    return {"n": n, "W": W, "L": L, "D": D, "rate": rate, "b8_illegal": b8_illegal, "b8_cue8": b8_cue8}

by_variant = defaultdict(list)
for p in glob.glob(os.path.join(log_dir, "*.jsonl")):
    base = os.path.basename(p)
    variant = base.split("_seed", 1)[0]
    s = summarize(p)
    if s["n"] == 0:
        continue
    by_variant[variant].append(s)

def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

rows = []
for variant, ss in by_variant.items():
    rates = [s["rate"] for s in ss]
    b8 = [s["b8_illegal"] + s["b8_cue8"] for s in ss]
    rows.append(
        (
            variant,
            len(ss),
            mean(rates),
            mean(b8),
            mean([s["b8_illegal"] for s in ss]),
            mean([s["b8_cue8"] for s in ss]),
        )
    )

rows.sort(key=lambda r: r[0])
print("\n=== ablation summary (mean over runs) ===")
print(f"log_dir={log_dir}")
print("variant,runs,win_rate_mean,black8_mean,illegal8_mean,cue+8_mean")
for r in rows:
    print(f"{r[0]},{r[1]},{r[2]:.3f},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f}")
PY

