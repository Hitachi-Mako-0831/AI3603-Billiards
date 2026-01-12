Param(
  [ValidateSet("basic", "pro")]
  [string]$Opponent = $(if ($env:OPPONENT) { $env:OPPONENT } else { "basic" }),
  [int]$NGames = $(if ($env:N_GAMES) { [int]$env:N_GAMES } else { 120 }),
  [int]$Repeats = $(if ($env:REPEATS) { [int]$env:REPEATS } else { 3 }),
  [int]$SeedBase = $(if ($env:SEED_BASE) { [int]$env:SEED_BASE } else { 42 }),
  [bool]$FixedSeed = $(if ($env:FIXED_SEED) { [bool]([int]$env:FIXED_SEED) } else { $true }),
  [string]$LogDir = $(if ($env:LOG_DIR) { $env:LOG_DIR } else { (Join-Path "logs" (Join-Path "ablations" $Opponent)) })
)

$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

function Enable-Conda {
  if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    return
  }
  try {
    (& conda "shell.powershell" "hook") | Out-String | Invoke-Expression
  } catch {
    return
  }
  try {
    conda activate billiards_rl | Out-Null
  } catch {
    return
  }
}

Enable-Conda

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Run-Variant {
  Param(
    [string]$Name,
    [string[]]$ExtraArgs
  )

  for ($r = 0; $r -lt $Repeats; $r++) {
    $seed = $SeedBase + $r
    $logPath = Join-Path $LogDir ("{0}_seed{1}.jsonl" -f $Name, $seed)
    if (Test-Path $logPath) {
      Remove-Item -Force $logPath
    }

    $args = @("--opponent", $Opponent, "--n_games", "$NGames", "--log_path", $logPath)
    if ($FixedSeed) {
      $args += @("--fixed_seed", "--seed", "$seed")
    }
    if ($ExtraArgs) {
      $args += $ExtraArgs
    }

    python evaluate.py @args
    python analyze_eval_log.py --log $logPath --our PhysicsAgent --limit_illegal_print 0
  }
}

Run-Variant -Name "full" -ExtraArgs @()
Run-Variant -Name "no_safety_gate" -ExtraArgs @("--no_safety_gate")
Run-Variant -Name "no_robust_rollout" -ExtraArgs @("--no_robust_rollout")
Run-Variant -Name "no_risk_shaping" -ExtraArgs @("--no_risk_shaping")
Run-Variant -Name "no_fallback" -ExtraArgs @("--no_fallback")

$env:LOG_DIR = $LogDir
$env:OPPONENT = $Opponent

@'
import glob
import json
import os
from collections import defaultdict

log_dir = os.environ.get("LOG_DIR")
opponent = os.environ.get("OPPONENT", "basic")
if not log_dir:
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
'@ | python -

