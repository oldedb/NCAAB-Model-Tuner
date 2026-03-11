# NCAAB Auto-Research Score Prediction

## Project Overview

Autonomous, self-improving college basketball score prediction system based on Karpathy's "autoresearch" pattern. An AI agent iteratively improves a prediction model that forecasts actual NCAA D1 men's basketball game scores, optimizing against a single metric (MAE).

**Primary goal**: Learn and apply the Karpathy auto-research loop pattern. The loop is the skill — the basketball prediction is the test case.

## Architecture — Three Files That Matter

### `prepare.py` — Fixed Infrastructure (DO NOT MODIFY)
- Loads raw game data from ESPN (data/raw_games.csv)
- Computes season-to-date features for each team (point-in-time, no leakage)
- Splits data into training and validation sets
- Runs evaluation and outputs MAE score
- Sanity checks: predictions clamped to 40-120 points
- **The agent NEVER touches this file.**

### `predict.py` — Prediction Model (AGENT MODIFIES THIS ONLY)
- The ONE file the agent edits during the auto-research loop
- Contains: feature selection, model choice, hyperparameters, weighting, home court logic
- Everything here is fair game: algorithm, features, math, approach

### `program.md` — Strategy Document (HUMAN WRITES THIS)
- Plain-English guidance for the agent's research direction
- Basketball domain knowledge, handicapping instincts, what to try/avoid
- Human iterates on this based on experiment results

## Scoring Metric

**Primary: Mean Absolute Error (MAE)** — Lower is better.
- For each game: |Predicted Score - Actual Score| for both teams
- MAE = average of all errors across all games and both teams
- Example: Predict Duke 78 UNC 72, actual Duke 75 UNC 70 → errors 3 and 2 → avg 2.5

**Secondary (logged only, not for keep/discard):**
- Margin MAE (predicted margin vs actual margin)
- Total MAE (predicted total vs actual total)
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

**ESPN public scoreboard API** — no API key required.

### Features Available (pre-computed in prepare.py, prefixed home_/away_)
All features are season-to-date as of game day (point-in-time, no leakage):
- `ppg` — points per game
- `opp_ppg` — opponent PPG (defensive quality)
- `win_pct` — winning percentage
- `avg_margin` — average scoring margin
- `games_played` — games played so far this season
- `last5_ppg` / `last5_opp_ppg` / `last5_margin` — last 5 games (recent form)
- `home_ppg` / `away_ppg` — scoring by venue
- `days_rest` — days since last game

Also: `neutral_site` (bool)

### Data Pipeline
ESPN scoreboard API → raw_games.csv → prepare.py computes features → predict.py uses them

### Train/Validation Split
- Training: seasons 2022-2024
- Validation: season 2025

## Project Structure

```
NCAAB-Model-Tuner/
├── CLAUDE.md                              # This file — project instructions
├── NCAAB_AutoResearch_Scope_of_Work.md    # Original scope of work reference
├── program.md                             # Strategy doc (human-written)
├── prepare.py                             # Fixed eval infrastructure
├── predict.py                             # Model code (agent-editable)
├── data/
│   └── raw_games.csv                      # ESPN game results
├── logs/                                  # Experiment logs
├── scripts/
│   └── pull_data.py                       # ESPN data puller
├── .env                                   # API keys (not committed)
├── .gitignore
└── requirements.txt
```

## Implementation Phases

1. **Data Acquisition** — Pull game scores from ESPN API (seasons 2022-2026)
2. **Baseline Model** — Verify end-to-end pipeline, establish baseline MAE
3. **Strategy & Loop** — Write `program.md`, build loop runner script
4. **First Autonomous Run** — Run loop, review results, iterate on `program.md`
5. **Refinement** — Expand features, tune strategy doc, explore ensembles

## Technical Stack

- **Python 3.10+** — prediction loop, ML models
- **Git** — experiment tracking (feature branch for experiments)
- **Key Python libs:** pandas, numpy, scikit-learn, xgboost

## Success Criteria

| Level | Target |
|-------|--------|
| Minimum | Loop runs autonomously, MAE improves from baseline |
| Good | MAE consistently < 9 points per team per game |
| Excellent | MAE < 7 points, directional accuracy > 70% |
| Stretch | Nightly runs, daily predictions competitive with public models |

## Guard Rails

- Agent only modifies `predict.py` during the auto-research loop
- Predictions must be 40-120 points (enforced in `prepare.py`)
- Each experiment has a fixed time budget
- All successful changes are git-committed with descriptive messages
- Failed experiments are logged but reverted
