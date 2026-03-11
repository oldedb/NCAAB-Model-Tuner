#!/usr/bin/env bash
#
# Auto-Research Loop Runner
# Runs the Karpathy-style experiment loop:
#   1. Agent modifies predict.py
#   2. Evaluate against validation set
#   3. Keep if MAE improves, revert if not
#   4. Repeat
#
# Usage:
#   ./scripts/run_loop.sh [max_experiments]
#   Default: 50 experiments

set -uo pipefail

# Allow launching Claude from within an existing session
unset CLAUDECODE 2>/dev/null || true

cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Full paths to avoid PATH issues
CLAUDE="$HOME/.local/bin/claude"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

MAX_EXPERIMENTS=${1:-50}
LOG_DIR="$PROJECT_ROOT/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/session_${TIMESTAMP}.log"
PREDICT_FILE="$PROJECT_ROOT/predict.py"
BEST_PREDICT="$LOG_DIR/best_predict.py"

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# --- Get current MAE score ---
get_score() {
    "$PYTHON" "$PROJECT_ROOT/prepare.py" 2>&1 | tee -a "$LOG_FILE" | grep "^SCORE:" | awk '{print $2}'
}

# --- Initialize ---
log "=========================================="
log "NCAAB Auto-Research Loop"
log "Max experiments: $MAX_EXPERIMENTS"
log "=========================================="

# Verify tools exist
if [ ! -f "$CLAUDE" ]; then
    log "ERROR: claude not found at $CLAUDE"
    exit 1
fi
if [ ! -f "$PYTHON" ]; then
    log "ERROR: python not found at $PYTHON"
    exit 1
fi

# Establish baseline
log ""
log "--- Establishing baseline ---"
BEST_SCORE=$(get_score)

if [ -z "$BEST_SCORE" ]; then
    log "ERROR: Could not get baseline score. Check prepare.py and data."
    exit 1
fi

log "Baseline MAE: $BEST_SCORE"

# Save current predict.py as best
cp "$PREDICT_FILE" "$BEST_PREDICT"

# Track stats
KEPT=0
DISCARDED=0

# --- Main loop ---
for i in $(seq 1 "$MAX_EXPERIMENTS"); do
    log ""
    log "=========================================="
    log "EXPERIMENT $i / $MAX_EXPERIMENTS"
    log "Best MAE so far: $BEST_SCORE (kept: $KEPT, discarded: $DISCARDED)"
    log "=========================================="

    # Save pre-experiment version
    cp "$PREDICT_FILE" "$LOG_DIR/predict_before_exp_${i}.py"

    # Ask Claude to modify predict.py
    log "Asking agent to modify predict.py..."

    PROMPT="You are running experiment $i of $MAX_EXPERIMENTS in an auto-research loop.
Working directory: $PROJECT_ROOT

CURRENT BEST MAE: $BEST_SCORE
EXPERIMENTS SO FAR: kept=$KEPT, discarded=$DISCARDED

Read program.md for strategy guidance, then read the current predict.py.

Make ONE focused modification to predict.py to try to lower the MAE score.
The modification should be a single, testable change — not a complete rewrite.

Ideas for changes:
- Try a different model (linear regression, ridge, random forest, xgboost)
- Add or remove features
- Change feature engineering (interactions, normalizations, transformations)
- Adjust home court advantage logic
- Try predicting margin + total instead of individual scores
- Change how recent form is weighted vs season averages
- Add sample size regression (blend with league avg for small games_played)

IMPORTANT:
- Only modify predict.py
- The predict() function signature must stay the same: predict(train_df, val_df) -> DataFrame with pred_home_score, pred_away_score
- Keep the code clean and runnable
- Do NOT add any print statements or logging
- After editing, do NOT run prepare.py — the loop runner handles evaluation

Write a brief comment at the top of predict.py describing what this experiment changes."

    "$CLAUDE" --dangerously-skip-permissions -p "$PROMPT" >> "$LOG_FILE" 2>&1
    CLAUDE_EXIT=$?

    if [ $CLAUDE_EXIT -ne 0 ]; then
        log "Agent errored (exit: $CLAUDE_EXIT). Reverting."
        cp "$BEST_PREDICT" "$PREDICT_FILE"
        DISCARDED=$((DISCARDED + 1))
        continue
    fi

    # Evaluate the new predict.py
    log "Evaluating..."
    NEW_SCORE=$(get_score)

    if [ -z "$NEW_SCORE" ]; then
        log "ERROR: Evaluation failed. Reverting."
        cp "$BEST_PREDICT" "$PREDICT_FILE"
        DISCARDED=$((DISCARDED + 1))
        continue
    fi

    log "New MAE: $NEW_SCORE (was: $BEST_SCORE)"

    # Compare scores (lower is better)
    IS_BETTER=$(awk "BEGIN {print ($NEW_SCORE < $BEST_SCORE) ? 1 : 0}")

    if [ "$IS_BETTER" -eq 1 ]; then
        log "IMPROVEMENT! Keeping change. ($BEST_SCORE -> $NEW_SCORE)"
        BEST_SCORE="$NEW_SCORE"
        cp "$PREDICT_FILE" "$BEST_PREDICT"
        cp "$PREDICT_FILE" "$LOG_DIR/predict_best_exp_${i}_mae_${NEW_SCORE}.py"
        KEPT=$((KEPT + 1))
    else
        log "No improvement. Reverting. ($NEW_SCORE >= $BEST_SCORE)"
        cp "$BEST_PREDICT" "$PREDICT_FILE"
        DISCARDED=$((DISCARDED + 1))
    fi
done

# --- Summary ---
log ""
log "=========================================="
log "SESSION COMPLETE"
log "=========================================="
log "Total experiments: $MAX_EXPERIMENTS"
log "Kept:             $KEPT"
log "Discarded:        $DISCARDED"
log "Starting MAE:     (see baseline above)"
log "Final best MAE:   $BEST_SCORE"
log "Best predict.py:  $BEST_PREDICT"
log "Full log:         $LOG_FILE"
log "=========================================="
