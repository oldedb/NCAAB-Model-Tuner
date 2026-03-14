# Testing Ideas for program.md

## Revert here for original program.md file

```bash
git checkout a04466d -- program.md
```

This restores `program.md` to the version before any testing strategy was applied (MAE: 7.78, 95 features, generic guidance).

---

## Results Summary

| Strategy | Experiments | Kept | MAE Before | MAE After | Change |
|----------|------------|------|------------|-----------|--------|
| Feature Pruning | 25 | 0 | 7.7787 | 7.7787 | 0 |
| Focus on Margin Prediction | 25 | 2 | 7.7811 | 7.7787 | -0.0024 |
| Try Different Models | 25 | 1 | 7.7787 | 7.7768 | -0.0019 |
| Hyperparameter Sweep | 25 | 1 | 7.7768 | 7.7764 | -0.0004 |
| Fix Total Prediction | 25 | 0 | 7.7764 | 7.7764 | 0 |
| **Conference/SoS Adjustment** | **25** | **5** | **7.7764** | **7.7592** | **-0.0172** |
| Stacking / Meta-Model | 25 | 2 | 7.7592 | 7.7573 | -0.0019 |
| Shooting Matchup Deep Dive | 25 | 0 | 7.7573 | 7.7573 | 0 |
| Lean Into Power Ratings | 25 | 4 | 7.7573 | 7.7512 | -0.0061 |
| Reduce Overfitting | 25 | 0 | 7.7512 | 7.7512 | 0 |

**Total: 250 experiments, 15 kept, MAE 7.7787 → 7.7512**

Best strategy: Conference/SoS Adjustment (5 kept, -0.0172 MAE)

---

Copy any of these blocks into `program.md` under "Current Focus" to steer the next tuning run. Be specific — the more focused the guidance, the fewer wasted experiments.

---

## Feature Pruning (tested — no improvement)

> The model has 95 features but many have <1% importance. Focus the next run on removing weak features. Try dropping everything below 2% importance. Try Lasso for auto-selection. Do NOT add new features.

## Try Different Models (tested — added Random Forest, MAE -0.0019)

> XGBoost + Ridge is plateauing. Try LightGBM as a replacement for XGBoost. Try a 3-model ensemble (XGBoost + Ridge + Random Forest). Do NOT change the feature list.

## Fix Total Prediction (tested — no improvement)

> The model is underestimating high-scoring games and overestimating low-scoring games. Focus on improving the total prediction. Pace features should be weighted more heavily.

## Hyperparameter Sweep (tested — MAE -0.0004)

> Only change ONE hyperparameter per experiment. Do not rewrite the model architecture. Try: XGBoost max_depth 3 vs 4 vs 5, learning_rate 0.04 vs 0.06 vs 0.08, n_estimators 150 vs 200 vs 300.

## Lean Into Power Ratings (tested — MAE -0.0061)

> Elo and SRS are carrying 36% of the model's importance. Build more features from them: Elo momentum (change over last 5 games), SRS rank differential, conference-adjusted SRS. Do NOT remove existing power rating features.

## Focus on Margin Prediction (tested — MAE -0.0024)

> Margin MAE (8.33) is worse than score MAE (7.78). Focus experiments on improving the margin model specifically. Try separate feature sets for total vs margin. Try giving the margin model more Ridge weight since it benefits from regularization.

## Conference/Strength of Schedule Adjustment (tested — best strategy, MAE -0.0172)

> Add conference strength features. Average SRS by conference, then compute how far above/below conference average each team sits. Conference tournament games may behave differently than regular season — explore whether conference_game flag helps.

## Reduce Overfitting (tested — no improvement)

> The model may be overfitting with 95 features and max_depth=4. Try: reduce max_depth to 3, increase reg_lambda to 15 or 20, reduce colsample_bytree to 0.5, or use cross-validation within the training set to select hyperparameters.

## Shooting Matchup Deep Dive (tested — no improvement)

> Explore whether 3-point reliance matchups matter. A team that shoots 40% of their shots from 3 against a team with poor perimeter defense (high opp_efg_pct) should score more. Try interaction features: home_fg3_rate * away_opp_efg_pct.

## Recency Weighting (not yet tested)

> Season-to-date averages weight game 1 the same as game 25. Try exponentially weighted averages that favor recent games. Try separate models for early season (games_played < 10) vs late season.

## Stacking / Meta-Model (tested — MAE -0.0019)

> Instead of a fixed 75/25 or 55/45 blend, train a meta-model that learns the optimal blend. Use XGBoost and Ridge predictions as features for a simple linear meta-learner trained on a holdout fold.

---

## How to Use

1. Pick one or two ideas that match what you're seeing in the results
2. Paste them into `program.md` under "Current Focus"
3. Remove or comment out ideas you don't want the agent to pursue
4. Run the loop: `./scripts/run_loop.sh 50`
5. Review results, update strategy, repeat
