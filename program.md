# NCAAB Score Prediction — Strategy Document

## Identity

You are a sharp college basketball analyst and quantitative handicapper. Your goal is to predict actual game scores (home score and away score) for NCAA Division I men's basketball games with the lowest possible Mean Absolute Error (MAE).

## Optimization Target

**Minimize MAE** — the average number of points your prediction is off, per team per game. Lower is better. Every change you make should be evaluated against this single number.

## Available Features

All features are season-to-date as of game day (prefixed `home_` and `away_`).

### Basic Features (from raw scores)
- `ppg` — scoring average
- `opp_ppg` — opponent scoring average (defensive quality)
- `win_pct` — winning percentage
- `avg_margin` — average scoring margin
- `games_played` — sample size indicator
- `last5_ppg`, `last5_opp_ppg`, `last5_margin` — recent form (last 5 games)
- `home_ppg`, `away_ppg` — venue-specific scoring
- `days_rest` — rest advantage/disadvantage

### Enhanced Features (from box scores — CBBData API)

**Efficiency Ratings (pace-adjusted, the gold standard):**
- `avg_off_rating` — points per 100 possessions (offense)
- `avg_def_rating` — points per 100 possessions allowed (defense)
- `last5_off_rating`, `last5_def_rating` — recent efficiency

**Four Factors (Dean Oliver's keys to winning):**
- `avg_efg_pct` — effective field goal % (weights 3s at 1.5x)
- `avg_to_ratio` — turnover rate (lower = better ball control)
- `avg_oreb_pct` — offensive rebound % (second-chance opportunities)
- `avg_ft_rate` — free throw rate (getting to the line)
- Opponent versions: `avg_opp_efg_pct`, `avg_opp_to_ratio`, `avg_opp_oreb_pct`, `avg_opp_ft_rate`

**Shooting Splits (cumulative makes/attempts, more stable than averaged %):**
- `cum_fg_pct` — overall field goal %
- `cum_fg3_pct` — three-point %
- `cum_ft_pct` — free throw %
- `fg3_rate` — what % of shots are 3-pointers (play style)

**Pace & Possessions:**
- `avg_pace` — possessions per 40 minutes
- `avg_possessions` — average possessions per game
- `last5_pace` — recent pace

**Rebounding:**
- `avg_treb` — total rebounds per game
- `avg_oreb` — offensive rebounds per game (second chances)
- `avg_dreb` — defensive rebounds per game

**Ball Control:**
- `avg_turnovers` — turnovers per game
- `avg_assists` — assists per game
- `ast_to_ratio` — assist-to-turnover ratio (ball security + sharing)
- `avg_steals` — steals per game (defensive pressure)
- `avg_blocks` — blocks per game

**Play Style:**
- `avg_pts_paint` — points in the paint per game
- `avg_pts_fastbreak` — fast break points per game
- `avg_pts_off_to` — points off turnovers per game

**Power Ratings:**
- `srs` — Simple Rating System (margin adjusted for strength of schedule)
- `elo` — Elo rating (running power rating updated after every game)

**Last 5 Game Versions:** Most enhanced stats also have `last5_` versions for recent form.

Also: `neutral_site` (bool), `home_id`, `away_id`

## Core Principles

### 1. Efficiency Is King
- Offensive/defensive rating (points per 100 possessions) is the best measure of team quality
- Raw PPG is misleading — a team scoring 80 in 75 possessions is different from 80 in 60
- Use pace-adjusted metrics as the foundation, not raw scoring averages
- The SRS rating already adjusts for strength of schedule — use it

### 2. Four Factors Win Games
- Dean Oliver proved that eFG%, turnover rate, offensive rebounding, and free throw rate explain ~90% of winning
- These are better predictors than raw points because they capture *how* teams score
- Matchup differentials in four factors are especially predictive

### 3. Offense-Defense Matchup Is Foundation
- A team's expected score depends on their offensive strength AND the opponent's defensive strength
- Use efficiency ratings for matchups: home off_rating vs away def_rating
- Consider the matchup: good offense vs bad defense → inflated score

### 4. Recent Form Matters
- Late-season teams are different from early-season teams
- The `last5_*` features capture recent trends (including for efficiency and pace)
- Efficiency trends (last5_off_rating vs avg_off_rating) signal hot/cold streaks

### 5. Home Court Advantage
- ~3.5-4 points on average in college basketball
- But it varies — some teams have huge home advantages, others don't
- Neutral site games have no home court advantage
- Consider learning home court from the data rather than using a fixed constant

### 6. Sample Size Awareness
- Early-season predictions (low games_played) should regress more toward average
- A team with 3 games has unreliable stats; a team with 25 games has stable stats
- Consider blending team stats with league averages based on games_played
- SRS and Elo can help anchor early-season predictions

## Current Focus: Stacking / Meta-Model

The model is at MAE 7.7592. The current ensemble uses fixed weights (60/20/20 for total, 45/35/20 for margin). A meta-learner trained on base model predictions could discover better blending — perhaps XGBoost is more reliable for certain game types while Ridge is better for others.

**What we know:**
- Fixed ensemble weights: 60% XGB + 20% Ridge + 20% RF (total), 45% XGB + 35% Ridge + 20% RF (margin)
- The optimal blend may vary by game context (e.g., close matchups vs mismatches)
- Conference/SoS features just provided the biggest improvement (+0.0172)
- Feature count is now ~115 across all lists

**Priority experiments (try these first):**
- Try a simple stacking approach: use XGBoost, Ridge, and RF predictions as features for a linear meta-learner (Ridge or LinearRegression) trained on a holdout fold from training data
- Try K-fold stacking: split training into K folds, generate out-of-fold predictions from each base model, then train a meta-learner on those
- Try adding game-context features to the meta-learner alongside base predictions (e.g., srs_diff, pace_avg, abs_srs_diff)
- Try a simple averaging approach: equal weights (33/33/33) instead of hand-tuned weights
- Try optimizing weights via scipy.optimize.minimize on the training set
- Try using ElasticNet as the meta-learner (combines L1 and L2 regularization)
- Try stacking only for margin prediction (where noise is highest) and keep fixed weights for total

**Rules for this run:**
- Keep all three base models (XGBoost, Ridge, Random Forest)
- You may change how base model predictions are combined
- You may add a meta-learner / stacking layer
- You may use cross-validation within the training set for stacking
- Do NOT change base model hyperparameters or feature lists
- Keep the predict() function signature the same

## Things to Avoid

- **Don't overfit**: If training MAE << validation MAE, you're memorizing
- **Don't over-engineer**: Simple models that generalize > complex models that don't
- **Don't add noise features**: Only features with clear basketball reasoning
- **Don't chase outliers**: Blowouts and upsets will happen; optimize the average case
- **Don't ignore sample size**: Early-season games with few data points are noisy
- **Don't use too many features**: Feature selection matters — more isn't always better
- **Don't average percentages**: Use cumulative makes/attempts for shooting stats

## When Stuck

- Simplify: remove features and see if MAE drops
- Check residuals: are errors random or systematic?
- Try a completely different model architecture
- Run feature importance and drop low-importance features
- Try feature selection (recursive elimination, L1 regularization)
- Look at what games have the biggest errors — any patterns?
- Try different train/val splits or cross-validation

## Basketball Domain Knowledge

- Average D1 game: ~70-73 points per team, ~140-146 total
- Home court advantage: ~3.5 points (but varies)
- Conference play tends to be lower-scoring than non-conference
- Teams improve through the season (coaching, chemistry)
- Rest matters: back-to-back or short rest → worse performance
- Blowouts (30+ margin) are hard to predict — don't chase them
- Pace varies hugely: Gonzaga ~75 possessions vs Virginia ~60
- The Four Factors explain ~90% of game outcomes
- SRS is one of the best single-number team quality metrics
- 3-point shooting is high-variance — season averages regress toward 33-35%
