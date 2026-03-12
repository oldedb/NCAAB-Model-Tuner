# NCAAB Model Tuner

A self-improving college basketball score prediction system built on Karpathy's "autoresearch" pattern. An AI agent autonomously experiments with prediction models, keeping changes that improve accuracy and reverting those that don't.

## How It Works

The system has three parts:

1. **Fixed infrastructure** (`prepare.py`) — Loads game data, computes team stats, and evaluates predictions. Never modified by the agent.
2. **Prediction model** (`predict.py`) — The ONE file the AI agent modifies. Contains the model architecture, features, and training logic.
3. **Strategy document** (`program.md`) — Human-written guidance that tells the agent what to try next.

The auto-research loop (`scripts/run_loop.sh`) ties them together:
- Ask the AI agent to make one change to `predict.py`
- Evaluate the new version against historical data
- If MAE (Mean Absolute Error) improves → keep the change
- If not → revert and try again

After 50 experiments, the model improved from **8.61 MAE** (baseline) to **8.44 MAE** (current best).

## Current Model

The tuned model uses a 70/30 XGBoost + Ridge regression ensemble that:
- Predicts game **total** and **margin** separately, then derives individual scores
- Uses 21 base features and 21 engineered features per game
- Applies Bayesian shrinkage to handle teams with few games played
- Features include matchup differentials, pace proxies, momentum trends, and rest advantage

## Predict Upcoming Games

To predict scores for upcoming games:

```bash
# Tomorrow's games
python predict_tomorrow.py

# Today's games
python predict_tomorrow.py today

# A specific date
python predict_tomorrow.py 2026-03-15
```

The script fetches the schedule from ESPN, computes current team stats from this season's results, and runs them through the trained model.

## Setup

### Requirements

- Python 3.10+
- ~19,000 historical game results (included in `data/raw_games.csv` after first data pull)

### Install

```bash
# Clone the repo
git clone https://github.com/oldedb/NCAAB-Model-Tuner.git
cd NCAAB-Model-Tuner

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Pull Game Data

```bash
python scripts/pull_data.py
```

This fetches game results from ESPN's public API for seasons 2022–2026. Takes a few minutes due to rate limiting.

### Run Predictions

```bash
# Evaluate model accuracy against validation data
python prepare.py

# Predict upcoming games
python predict_tomorrow.py
```

### Run the Auto-Research Loop

```bash
# Requires Claude Code CLI installed
./scripts/run_loop.sh        # default: 50 experiments
./scripts/run_loop.sh 10     # or specify a number
```

## Project Structure

```
├── prepare.py              # Data loading, feature computation, evaluation (fixed)
├── predict.py              # Prediction model (agent-modified)
├── predict_tomorrow.py     # Predict upcoming games from ESPN schedule
├── program.md              # Strategy guidance for the AI agent
├── requirements.txt        # Python dependencies
├── scripts/
│   ├── pull_data.py        # ESPN data acquisition
│   └── run_loop.sh         # Auto-research loop runner
├── data/
│   └── raw_games.csv       # Historical game results (gitignored)
└── logs/                   # Experiment logs and snapshots (gitignored)
```

## Data Source

All game data comes from ESPN's public scoreboard API (no API key required). Features are computed from raw box scores — season-to-date averages calculated before each game to prevent data leakage.

## Inspired By

[Andrej Karpathy's "autoresearch" concept](https://x.com/karpathy/status/1886192184808149383) — using AI agents in a tight experiment loop with automatic evaluation and version control.
