# Testing Ideas for program.md

## Revert here for original program.md file

```bash
git checkout a04466d -- program.md
```

This restores `program.md` to the version before any testing strategy was applied (MAE: 7.78, 95 features, generic guidance).

---

Copy any of these blocks into `program.md` under "Approaches to Try" to steer the next tuning run. Be specific — the more focused the guidance, the fewer wasted experiments.

---

## Feature Pruning

> The model has 95 features but many have <1% importance. Focus the next run on removing weak features. Try dropping everything below 2% importance. Try Lasso for auto-selection. Do NOT add new features.

## Try Different Models

> XGBoost + Ridge is plateauing. Try LightGBM as a replacement for XGBoost. Try a 3-model ensemble (XGBoost + Ridge + Random Forest). Do NOT change the feature list.

## Fix Total Prediction (over/under issues)

> The model is underestimating high-scoring games and overestimating low-scoring games. Focus on improving the total prediction. Pace features should be weighted more heavily.

## Hyperparameter Sweep (safe, incremental)

> Only change ONE hyperparameter per experiment. Do not rewrite the model architecture. Try: XGBoost max_depth 3 vs 4 vs 5, learning_rate 0.04 vs 0.06 vs 0.08, n_estimators 150 vs 200 vs 300.

## Lean Into Power Ratings

> Elo and SRS are carrying 36% of the model's importance. Build more features from them: Elo momentum (change over last 5 games), SRS rank differential, conference-adjusted SRS. Do NOT remove existing power rating features.

## Focus on Margin Prediction

> Margin MAE (8.33) is worse than score MAE (7.78). Focus experiments on improving the margin model specifically. Try separate feature sets for total vs margin. Try giving the margin model more Ridge weight since it benefits from regularization.

## Conference/Strength of Schedule Adjustment

> Add conference strength features. Average SRS by conference, then compute how far above/below conference average each team sits. Conference tournament games may behave differently than regular season — explore whether conference_game flag helps.

## Reduce Overfitting

> The model may be overfitting with 95 features and max_depth=4. Try: reduce max_depth to 3, increase reg_lambda to 15 or 20, reduce colsample_bytree to 0.5, or use cross-validation within the training set to select hyperparameters.

## Shooting Matchup Deep Dive

> Explore whether 3-point reliance matchups matter. A team that shoots 40% of their shots from 3 against a team with poor perimeter defense (high opp_efg_pct) should score more. Try interaction features: home_fg3_rate * away_opp_efg_pct.

## Recency Weighting

> Season-to-date averages weight game 1 the same as game 25. Try exponentially weighted averages that favor recent games. Try separate models for early season (games_played < 10) vs late season.

## Stacking / Meta-Model

> Instead of a fixed 75/25 or 55/45 blend, train a meta-model that learns the optimal blend. Use XGBoost and Ridge predictions as features for a simple linear meta-learner trained on a holdout fold.

---

## How to Use

1. Pick one or two ideas that match what you're seeing in the results
2. Paste them into `program.md` under "Approaches to Try"
3. Remove or comment out ideas you don't want the agent to pursue
4. Run the loop: `./scripts/run_loop.sh 50`
5. Review results, update strategy, repeat
