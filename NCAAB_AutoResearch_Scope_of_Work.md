# NCAAB Auto-Research Score Prediction — Scope of Work

## Project Summary

Build an autonomous, self-improving college basketball score prediction system inspired by Andrej Karpathy's "autoresearch" framework. The system uses an AI agent running in a continuous loop to iteratively improve a prediction model that forecasts actual game scores (not just win/loss) for NCAA Division I men's basketball games.

The human writes a strategy document. The AI agent writes and refines the prediction code. A single, clear scoring metric decides what's working and what isn't. The loop runs overnight and accumulates improvements automatically.

---

## Background & Inspiration

### Karpathy's Auto-Research Pattern

Andrej Karpathy released a project called "autoresearch" (github.com/karpathy/autoresearch) that automates machine learning research using an agentic loop. The core pattern has three pieces:

1. **Fixed infrastructure** — data loading, evaluation, and scoring that does NOT change
2. **Modifiable prediction/training code** — the ONE file the AI agent is allowed to edit
3. **Strategy document (program.md)** — a plain-English markdown file written by the human that tells the agent how to approach the problem, what to try, what to avoid, and what "better" means

The agent reads the strategy doc, makes a change to the prediction code, runs an experiment, checks if the score improved, keeps or discards the change, and repeats indefinitely.

### Why This Fits NCAAB Score Prediction

This use case checks all five requirements for an effective agentic loop:

- **Scorable**: Prediction error (how far off were the predicted scores from actual scores) is a clean, objective number
- **Fast iterations**: Running predictions against historical games takes seconds, not hours
- **Bounded workspace**: The agent only modifies the prediction model logic
- **Low cost of failure**: Bad predictions against historical data waste compute time, not money
- **Traceable**: Every experiment is git-committed with its score, building a full history

---

## Data Source: CBBData + Barttorvik

### Why CBBData

CBBData (cbbdata.aweatherman.com) is a free R package that provides programmatic access to Barttorvik (T-Rank) data. It replaces the older toRvik package. Key advantages:

- **Free API key** — no subscription required for Barttorvik data
- **Historical ratings by date** — archived T-Rank ratings back to the 2014-15 season, aggregated by date, team, or year. This is critical for getting point-in-time snapshots (what were the ratings BEFORE a game was played)
- **Game-by-game factors** — adjusted offensive and defensive efficiency for each game with dates
- **Game predictions** — Barttorvik's own predictor model returns expected possessions, points, and win percentage for any game on any date at any location back to 2015
- **Updates every 15 minutes** during the season
- **Nearly 30 endpoints** covering player stats, team analytics, game results, and advanced metrics

### Key Data Elements Needed

For each historical game, the dataset should contain (at minimum):

**Pre-game ratings (as of the day before or morning of the game):**
- Team A: AdjOE (Adjusted Offensive Efficiency), AdjDE (Adjusted Defensive Efficiency), Barthag, AdjTempo
- Team B: AdjOE, AdjDE, Barthag, AdjTempo
- eFG%, TOR (Turnover Rate), TORD (Turnover Rate Defensive), ORB%, DRB%, FTR (Free Throw Rate)
- Any other available Four Factors or advanced metrics

**Game context:**
- Date
- Location (home/away/neutral)
- Conference for each team

**Actual outcome:**
- Team A final score
- Team B final score

### Data Note: Point-in-Time Ratings

This is the single most important data integrity requirement. The model MUST be trained on ratings as they existed BEFORE each game, not current/end-of-season ratings applied retroactively. Using current ratings to "predict" past games is data leakage — the ratings already absorbed those game results. CBBData's historical archive capability (by date) solves this.

### Data Pull Strategy

CBBData is an R package. The data pull phase will:

1. Use R to pull historical data from CBBData/Barttorvik endpoints
2. Export the combined dataset as CSV files
3. Those CSVs become the fixed input for the Python-based auto-research loop

This is a one-time (or periodic) data preparation step, similar to Karpathy's `prepare.py`.

---

## System Architecture

### Three Files That Matter

Following the auto-research pattern exactly:

#### 1. `prepare.py` — Fixed Infrastructure (DO NOT MODIFY)

This file handles:
- Loading the historical game dataset (CSV files from CBBData export)
- Splitting data into training set and validation set (e.g., use seasons 2015-2024 for training, 2025 season for validation — or a rolling split)
- Defining the evaluation function that scores predictions against actual outcomes
- Outputting the single scoring metric after each run

**The agent never touches this file.**

#### 2. `predict.py` — The Prediction Model (AGENT MODIFIES THIS)

This is the ONE file the AI agent is allowed to edit. It contains:
- Feature engineering (which stats to use, how to combine them, any transformations)
- The prediction model itself (could be linear regression, XGBoost, neural net, ensemble — the agent decides)
- Weighting logic (how much to weight different factors)
- Home court advantage adjustments
- Tempo-based scoring projections
- Any other prediction logic

Everything in this file is fair game for the agent to modify: the algorithm, the features, the hyperparameters, the math, the approach.

#### 3. `program.md` — Strategy Document (HUMAN WRITES THIS)

This is the plain-English markdown file that tells the agent how to behave as a researcher. This is where the human's basketball knowledge and handicapping instincts live. Example contents:

- "You are a sharp college basketball analyst trying to predict actual game scores"
- "Prioritize AdjOE/AdjDE efficiency differentials as your foundation"
- "Pace/tempo is essential — a game between two 65-possession teams will have a fundamentally different total than two 75-possession teams"
- "Home court advantage matters but varies by team and conference"
- "Recent form should be weighted more heavily than early-season results"
- "Don't over-index on outlier blowouts"
- "Try combining efficiency metrics with tempo to project possessions first, then points per possession, then total points"
- "Be aggressive about testing non-obvious feature interactions"
- "If stuck, try simplifying the model rather than adding complexity"

**The human iterates on this file over time based on what the agent is (or isn't) finding.**

---

## Scoring Metric

The system needs a single, unambiguous number that tells the agent whether its latest change made predictions better or worse. Lower is better.

### Primary Metric: Mean Absolute Error (MAE) on Predicted Scores

For each game in the validation set:
- The model predicts Team A score and Team B score
- Compare to actual Team A score and actual Team B score
- Error = |Predicted Score - Actual Score| for each team in each game
- MAE = average of all errors across all games and both teams

**Example**: If the model predicts Duke 78, UNC 72 and the actual score is Duke 75, UNC 70, the errors are 3 and 2, averaging to 2.5 for that game.

### Why MAE

- Simple to understand (average number of points the prediction is off)
- Directly meaningful in a basketball context
- Not distorted by a few huge misses the way squared error can be
- Easy for the agent to compare: "My MAE went from 8.2 to 7.9 — that's an improvement, keep it"

### Optional Secondary Metrics (Logged but NOT used for keep/discard decisions)

- Spread accuracy: How often would the predicted margin have beaten the closing spread?
- Total accuracy: How close is the predicted total (Team A + Team B) to the actual total?
- Directional accuracy: What percentage of games did the model pick the correct winner?

These are logged for human review but the agent only optimizes against MAE.

---

## The Loop

### How One Experiment Works

1. Agent reads `program.md` for strategic guidance
2. Agent looks at current state of `predict.py`
3. Agent decides on a modification (new feature, different algorithm, changed weighting, etc.)
4. Agent edits `predict.py`
5. System runs the prediction model against the full validation set
6. `prepare.py` evaluates predictions and produces the MAE score
7. **If MAE is lower than previous best**: change is KEPT, committed to git branch, becomes new baseline
8. **If MAE is same or higher**: change is DISCARDED, code reverts to previous best version
9. Loop repeats from step 1

### Time Budget

Each experiment should complete in a fixed time window (e.g., 2-5 minutes depending on dataset size and model complexity). This ensures:
- All experiments are compared on equal footing
- The agent can't "cheat" by running a massively complex model that takes an hour
- You get ~12-30 experiments per hour
- An overnight run yields ~100-200 experiments

### Session Output

After a session, you have:
- A git log showing every successful improvement with commit messages describing what changed
- The current best MAE score
- The current best version of `predict.py`
- A log of all experiments (including discarded ones) showing what was tried

---

## Implementation Phases

### Phase 1: Data Acquisition & Preparation
- Set up R environment with CBBData package
- Pull historical Barttorvik ratings with point-in-time daily snapshots (2015-present)
- Pull game results with actual scores
- Join ratings to games (each game gets the ratings as of the day before)
- Export as clean CSV files
- Build `prepare.py` with data loading, train/validation split, and MAE evaluation function

### Phase 2: Baseline Prediction Model
- Build initial `predict.py` with a simple baseline (e.g., predict scores using AdjOE, AdjDE, and tempo)
- Verify the full pipeline works end-to-end: load data → predict → evaluate → output MAE
- Establish baseline MAE score

### Phase 3: Strategy Document & Loop Setup
- Write initial `program.md` with basketball handicapping knowledge and research guidance
- Set up the agentic loop infrastructure (shell script or Python wrapper that runs the agent, tracks scores, manages git commits)
- Configure Claude Code (or chosen agent) to operate within the repo

### Phase 4: First Autonomous Run
- Kick off the loop and let it run for a few hours
- Review results: what did the agent try? What worked? What didn't?
- Iterate on `program.md` based on observations

### Phase 5: Refinement & Expansion
- Iterate on the strategy document based on what the agent is finding
- Consider adding more data features (player-level stats, conference strength, rest days, etc.)
- Consider expanding the validation approach (rolling windows, out-of-sample testing on current season)
- Explore ensemble approaches where multiple loops run different strategies

---

## Technical Requirements

### Environment
- Python 3.10+ for the prediction loop
- R 4.x for CBBData data pulls (one-time or periodic)
- Git for version control and experiment tracking
- Claude Code (or equivalent AI coding agent) for the autonomous loop

### Python Dependencies (predict.py may use)
- pandas (data manipulation)
- numpy (numerical operations)
- scikit-learn (ML models — linear regression, random forest, etc.)
- xgboost / lightgbm (gradient boosting models, optional)
- Any other standard ML/stats libraries the agent chooses to try

### R Dependencies (data pull only)
- cbbdata
- dplyr
- readr (CSV export)

### Hardware
- No GPU required — this is tabular prediction, not neural network training
- Any modern laptop or desktop is sufficient
- Cloud compute optional but not necessary

---

## Key Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Data leakage (using future info to predict past) | Strict point-in-time rating snapshots; validation set is chronologically AFTER training set |
| Overfitting to historical patterns | Hold out most recent season entirely; monitor if validation MAE diverges from training MAE |
| Agent gets stuck in local optimum | Strategy doc encourages bold experimentation; periodic "resets" to try completely different approaches |
| CBBData API changes or goes down | Cache all pulled data as local CSV files immediately |
| Agent makes nonsensical predictions (negative scores, 200-point games) | Add sanity checks in prepare.py: predictions must be between 40-120 points |

---

## Success Criteria

- **Minimum viable**: The loop runs autonomously, produces experiments, and the MAE improves from baseline
- **Good**: MAE consistently below 8 points per team per game on validation data
- **Excellent**: MAE below 6 points per team per game, with predicted margins that would have beaten closing spreads at >53% rate
- **Stretch**: System runs nightly during the season, incorporating updated ratings, and produces daily game predictions that are competitive with Barttorvik's own projections

---

## What the Human Does vs. What the Agent Does

| Human | Agent |
|-------|-------|
| Writes and refines `program.md` | Reads `program.md` for guidance |
| Pulls and prepares the data (Phase 1) | Edits `predict.py` with new approaches |
| Reviews experiment results each morning | Runs experiments autonomously |
| Decides if the strategy doc needs updating | Decides what to try next based on strategy doc |
| Judges whether secondary metrics look promising | Optimizes against MAE score |
| Sets guard rails and sanity checks | Operates within guard rails |
| Brings basketball domain knowledge | Brings computational speed and tireless iteration |

---

## Notes for Claude Code Setup

- This document should live in the project root as the primary reference
- The agent should be pointed to `program.md` to begin each session
- All permissions for file modification should be limited to `predict.py` only
- Git should be initialized with a feature branch for experiments
- A shell script or loop wrapper should handle: running the agent → running predict.py → running evaluation → checking score → committing or reverting → restarting the agent with fresh context
