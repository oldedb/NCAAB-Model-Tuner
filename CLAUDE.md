# NCAAB Auto-Research Score Prediction

## Project Overview

Autonomous, self-improving college basketball score prediction system based on Karpathy's "autoresearch" pattern. An AI agent iteratively improves a prediction model that forecasts actual NCAA D1 men's basketball game scores, optimizing against a single metric (MAE).

## Architecture — Three Files That Matter

### `prepare.py` — Fixed Infrastructure (DO NOT MODIFY)
- Loads historical game data from CSV files (exported from CBBData/Barttorvik)
- Splits data into training and validation sets
- Runs the evaluation function and outputs the MAE score
- Contains sanity checks: predictions must be between 40-120 points
- **The agent NEVER touches this file.**

### `predict.py` — Prediction Model (AGENT MODIFIES THIS ONLY)
- The ONE file the agent is allowed to edit during the auto-research loop
- Contains: feature engineering, model selection, hyperparameters, weighting logic, home court adjustments, tempo projections
- Everything here is fair game: algorithm, features, math, approach

### `program.md` — Strategy Document (HUMAN WRITES THIS)
- Plain-English guidance for the agent's research direction
- Contains basketball domain knowledge, handicapping instincts, what to try/avoid
- Human iterates on this based on experiment results

## Scoring Metric

**Primary: Mean Absolute Error (MAE)** — Lower is better.
- For each game: |Predicted Score - Actual Score| for both teams
- MAE = average of all errors across all games and both teams
- Example: Predict Duke 78 UNC 72, actual Duke 75 UNC 70 → errors 3 and 2 → game avg 2.5

**Secondary (logged only, not used for keep/discard):**
- Spread accuracy (predicted margin vs closing spread, target >53%)
- Total accuracy (predicted total vs actual total)
- Directional accuracy (% correct winner picks)

## The Experiment Loop

1. Agent reads `program.md`
2. Agent reviews current `predict.py`
3. Agent makes a modification
4. System runs prediction against validation set
5. `prepare.py` evaluates and produces MAE
6. **MAE lower** → KEEP change, git commit, new baseline
7. **MAE same or higher** → DISCARD, revert to previous best
8. Repeat

Each experiment: 2-5 minute time budget. Overnight run yields ~100-200 experiments.

## Data Source

**CBBData** (R package) → Barttorvik/T-Rank data. Free API key required.

### Critical Rule: Point-in-Time Ratings
All ratings must be as they existed BEFORE each game — never end-of-season or current ratings applied retroactively. This prevents data leakage.

### Key Features
- AdjOE, AdjDE, Barthag, AdjTempo (for both teams)
- eFG%, TOR, TORD, ORB%, DRB%, FTR
- Game context: date, location (home/away/neutral), conference
- Actual scores (target variable)

### Data Pipeline
R pulls from CBBData → exports CSV → Python consumes CSV. One-time or periodic refresh.

## Project Structure

```
NCAAB-Model-Tuner/
├── CLAUDE.md                              # This file — project instructions
├── NCAAB_AutoResearch_Scope_of_Work.md    # Full scope of work reference
├── program.md                             # Strategy doc (human-written)
├── prepare.py                             # Fixed eval infrastructure
├── predict.py                             # Model code (agent-editable)
├── data/                                  # CSV files from CBBData export
│   └── *.csv
├── logs/                                  # Experiment logs
├── scripts/                               # Loop runner, data pull scripts
│   └── pull_data.R                        # R script for CBBData export
├── .env                                   # API keys (not committed)
├── .gitignore
└── requirements.txt
```

## Implementation Phases

1. **Data Acquisition** — Set up R + CBBData, pull historical ratings (2015-present) with point-in-time snapshots, export CSVs
2. **Baseline Model** — Build `prepare.py` and initial `predict.py`, verify end-to-end pipeline, establish baseline MAE
3. **Strategy & Loop** — Write `program.md`, build loop runner (shell script/wrapper), configure git branching
4. **First Autonomous Run** — Run loop for several hours, review results, iterate on `program.md`
5. **Refinement** — Expand features, tune strategy doc, explore ensembles, rolling validation

## Technical Stack

- **Python 3.10+** — prediction loop, ML models
- **R 4.x** — CBBData pulls only
- **Git** — experiment tracking (feature branch for experiments)
- **Key Python libs:** pandas, numpy, scikit-learn, xgboost/lightgbm (optional)
- **Key R libs:** cbbdata, dplyr, readr

## Success Criteria

| Level | Target |
|-------|--------|
| Minimum | Loop runs autonomously, MAE improves from baseline |
| Good | MAE consistently < 8 points per team per game |
| Excellent | MAE < 6 points, spread accuracy > 53% |
| Stretch | Nightly runs with updated ratings, competitive with Barttorvik projections |

## Guard Rails

- Agent only modifies `predict.py` during the auto-research loop
- Predictions must be 40-120 points (enforced in `prepare.py`)
- Each experiment has a fixed time budget
- All successful changes are git-committed with descriptive messages
- Failed experiments are logged but reverted
