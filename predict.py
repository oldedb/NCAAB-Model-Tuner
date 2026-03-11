"""
Prediction Model — The ONE file the AI agent modifies.

Experiment 21: Add form trend features (last5 - season avg) for offense and defense.
  Captures whether a team is trending up or down relative to their season baseline.
  home_off_trend = last5_ppg - ppg (positive = scoring more lately)
  home_def_trend = opp_ppg - last5_opp_ppg (positive = defending better lately)
  Also combined trend_diff features for matchup context.
  Previous best MAE: 8.4407
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


BASE_FEATURES = [
    "home_ppg", "home_opp_ppg", "home_avg_margin", "home_win_pct",
    "home_last5_ppg", "home_last5_opp_ppg", "home_last5_margin",
    "home_home_ppg", "home_days_rest", "home_games_played",
    "away_ppg", "away_opp_ppg", "away_avg_margin", "away_win_pct",
    "away_last5_ppg", "away_last5_opp_ppg", "away_last5_margin",
    "away_away_ppg", "away_days_rest", "away_games_played",
    "neutral_site",
]

ENGINEERED = [
    "home_off_vs_def",       # home offense vs away defense (raw)
    "away_off_vs_def",       # away offense vs home defense (raw)
    "margin_diff",           # difference in avg margins
    "recent_margin_diff",    # difference in recent form margins
    "home_reg_ppg",          # regression-adjusted PPG (home)
    "away_reg_ppg",          # regression-adjusted PPG (away)
    "home_reg_opp_ppg",      # regression-adjusted opp PPG (home)
    "away_reg_opp_ppg",      # regression-adjusted opp PPG (away)
    "reg_home_off_vs_def",   # regressed home offense vs regressed away defense
    "reg_away_off_vs_def",   # regressed away offense vs regressed home defense
    "rest_diff",             # home rest advantage
    "home_implied_pace",     # home team implied pace (ppg + opp_ppg)
    "away_implied_pace",     # away team implied pace
    "pace_diff",             # difference in implied pace
    "win_pct_diff",          # home win_pct - away win_pct
    "home_off_trend",        # home offensive momentum (last5_ppg - ppg)
    "away_off_trend",        # away offensive momentum
    "home_def_trend",        # home defensive momentum (opp_ppg - last5_opp_ppg)
    "away_def_trend",        # away defensive momentum
    "off_trend_diff",        # home off trend - away off trend
    "def_trend_diff",        # home def trend - away def trend
]

ALL_FEATURES = BASE_FEATURES + ENGINEERED


def _regress_to_mean(stat, games_played, league_avg, k=10):
    """Bayesian shrinkage: blend team stat toward league average based on sample size."""
    weight = games_played / (games_played + k)
    return weight * stat + (1 - weight) * league_avg


def _add_engineered(df, league_ppg=71.0, league_opp_ppg=71.0):
    """Add matchup-based and regression-adjusted engineered features."""
    df = df.copy()
    # How much home team scores vs how much away team allows
    df["home_off_vs_def"] = df["home_ppg"] - df["away_opp_ppg"]
    # How much away team scores vs how much home team allows
    df["away_off_vs_def"] = df["away_ppg"] - df["home_opp_ppg"]
    # Margin differential
    df["margin_diff"] = df["home_avg_margin"] - df["away_avg_margin"]
    # Recent form margin differential
    df["recent_margin_diff"] = df["home_last5_margin"] - df["away_last5_margin"]
    # Sample-size-regressed stats
    df["home_reg_ppg"] = _regress_to_mean(df["home_ppg"], df["home_games_played"], league_ppg)
    df["away_reg_ppg"] = _regress_to_mean(df["away_ppg"], df["away_games_played"], league_ppg)
    df["home_reg_opp_ppg"] = _regress_to_mean(df["home_opp_ppg"], df["home_games_played"], league_opp_ppg)
    df["away_reg_opp_ppg"] = _regress_to_mean(df["away_opp_ppg"], df["away_games_played"], league_opp_ppg)
    # Matchup differentials using regressed stats (more reliable with small samples)
    df["reg_home_off_vs_def"] = df["home_reg_ppg"] - df["away_reg_opp_ppg"]
    df["reg_away_off_vs_def"] = df["away_reg_ppg"] - df["home_reg_opp_ppg"]
    # Rest differential
    df["rest_diff"] = df["home_days_rest"] - df["away_days_rest"]
    # Implied pace (ppg + opp_ppg = total points in team's games)
    df["home_implied_pace"] = df["home_ppg"] + df["home_opp_ppg"]
    df["away_implied_pace"] = df["away_ppg"] + df["away_opp_ppg"]
    df["pace_diff"] = df["home_implied_pace"] - df["away_implied_pace"]
    # Win percentage differential (quality matchup)
    df["win_pct_diff"] = df["home_win_pct"] - df["away_win_pct"]
    # Form trends: last 5 games vs season average (momentum signal)
    df["home_off_trend"] = df["home_last5_ppg"] - df["home_ppg"]
    df["away_off_trend"] = df["away_last5_ppg"] - df["away_ppg"]
    df["home_def_trend"] = df["home_opp_ppg"] - df["home_last5_opp_ppg"]  # positive = better defense lately
    df["away_def_trend"] = df["away_opp_ppg"] - df["away_last5_opp_ppg"]
    df["off_trend_diff"] = df["home_off_trend"] - df["away_off_trend"]
    df["def_trend_diff"] = df["home_def_trend"] - df["away_def_trend"]
    return df


def predict(train_df, val_df):
    """
    Predict game scores for the validation set.

    Args:
        train_df: Historical training data (available for fitting models).
        val_df:   Games to predict with pre-game features for both teams.

    Returns:
        DataFrame with columns: pred_home_score, pred_away_score
        Must have the same index as val_df.
    """
    # Compute league averages from training data for regression
    league_ppg = train_df[["home_ppg", "away_ppg"]].mean().mean()
    league_opp_ppg = train_df[["home_opp_ppg", "away_opp_ppg"]].mean().mean()

    train = _add_engineered(train_df, league_ppg, league_opp_ppg)
    val = _add_engineered(val_df, league_ppg, league_opp_ppg)

    # Compute targets: total and margin
    train["total"] = train["home_score"] + train["away_score"]
    train["margin"] = train["home_score"] - train["away_score"]

    # Drop rows with missing features or targets
    train = train.dropna(subset=ALL_FEATURES + ["total", "margin"])

    X_train = train[ALL_FEATURES].astype(float)
    X_val = val[ALL_FEATURES].astype(float).fillna(X_train.median())

    # --- XGBoost predictions ---
    xgb_total = XGBRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=10.0,
        random_state=42, verbosity=0,
    )
    xgb_total.fit(X_train, train["total"])
    xgb_pred_total = xgb_total.predict(X_val)

    xgb_margin = XGBRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=10.0,
        random_state=42, verbosity=0,
    )
    xgb_margin.fit(X_train, train["margin"])
    xgb_pred_margin = xgb_margin.predict(X_val)

    # --- Ridge regression predictions ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    ridge_total = Ridge(alpha=100.0)
    ridge_total.fit(X_train_scaled, train["total"])
    ridge_pred_total = ridge_total.predict(X_val_scaled)

    ridge_margin = Ridge(alpha=100.0)
    ridge_margin.fit(X_train_scaled, train["margin"])
    ridge_pred_margin = ridge_margin.predict(X_val_scaled)

    # --- Ensemble: 70% XGBoost + 30% Ridge ---
    pred_total = 0.7 * xgb_pred_total + 0.3 * ridge_pred_total
    pred_margin = 0.7 * xgb_pred_margin + 0.3 * ridge_pred_margin

    # Derive individual scores from total and margin
    val["pred_home_score"] = (pred_total + pred_margin) / 2.0
    val["pred_away_score"] = (pred_total - pred_margin) / 2.0

    return val[["pred_home_score", "pred_away_score"]]
