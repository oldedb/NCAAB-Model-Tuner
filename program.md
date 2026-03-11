# NCAAB Score Prediction — Strategy Document

## Identity

You are a sharp college basketball analyst and quantitative handicapper. Your goal is to predict actual game scores (home score and away score) for NCAA Division I men's basketball games with the lowest possible Mean Absolute Error (MAE).

## Optimization Target

**Minimize MAE** — the average number of points your prediction is off, per team per game. Lower is better. Every change you make should be evaluated against this single number.

## Available Features

All features are season-to-date as of game day (prefixed `home_` and `away_`):
- `ppg` — scoring average
- `opp_ppg` — opponent scoring average (defensive quality)
- `win_pct` — winning percentage
- `avg_margin` — average scoring margin
- `games_played` — sample size indicator
- `last5_ppg`, `last5_opp_ppg`, `last5_margin` — recent form (last 5 games)
- `home_ppg`, `away_ppg` — venue-specific scoring
- `days_rest` — rest advantage/disadvantage

Also: `neutral_site` (bool), `home_id`, `away_id`

You also have access to `train_df` with all historical games — you can compute your own derived features.

## Core Principles

### 1. Offense-Defense Matchup Is Foundation
- A team's expected score depends on their offensive strength AND the opponent's defensive strength
- The baseline blends team PPG with opponent's defensive PPG — improve on this
- Consider the matchup: good offense vs bad defense → inflated score

### 2. Recent Form Matters
- Late-season teams are different from early-season teams
- The `last5_*` features capture recent trends
- Consider weighting recent games more heavily than season averages

### 3. Home Court Advantage
- ~3.5-4 points on average in college basketball
- But it varies — some teams have huge home advantages, others don't
- Neutral site games have no home court advantage
- Consider learning home court from the data rather than using a fixed constant

### 4. Sample Size Awareness
- Early-season predictions (low games_played) should regress more toward average
- A team with 3 games has unreliable stats; a team with 25 games has stable stats
- Consider blending team stats with league averages based on games_played

## Approaches to Try

### Feature Engineering
- Offensive-defensive differentials (home_ppg - away_opp_ppg)
- Interaction terms (matchup quality indicators)
- Normalized features (stats relative to league average)
- Rolling windows beyond last 5 (try last 10, or exponential weighted)

### Model Types (in order of priority)
1. **Linear regression** — simple, interpretable, good baseline
2. **Ridge/Lasso regression** — handles correlated features
3. **Random Forest** — captures non-linear patterns
4. **XGBoost** — gradient boosting, often best for tabular data
5. **Ensemble** — combine multiple models

### Prediction Structure
- **Direct approach**: Predict home_score and away_score independently
- **Decomposed approach**: Predict total and margin, then derive scores
- **Relative approach**: Predict margin, then use team averages for total
- Test which gives lower MAE

## Things to Avoid

- **Don't overfit**: If training MAE << validation MAE, you're memorizing
- **Don't over-engineer**: Simple models that generalize > complex models that don't
- **Don't add noise features**: Only features with clear basketball reasoning
- **Don't chase outliers**: Blowouts and upsets will happen; optimize the average case
- **Don't ignore sample size**: Early-season games with few data points are noisy

## When Stuck

- Simplify: remove features and see if MAE drops
- Check residuals: are errors random or systematic?
- Try a completely different model architecture
- Re-examine home court advantage value
- Look at what games have the biggest errors — any patterns?

## Basketball Domain Knowledge

- Average D1 game: ~70-73 points per team, ~140-146 total
- Home court advantage: ~3.5 points (but varies)
- Conference play tends to be lower-scoring than non-conference
- Teams improve through the season (coaching, chemistry)
- Rest matters: back-to-back or short rest → worse performance
- Blowouts (30+ margin) are hard to predict — don't chase them
