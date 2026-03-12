"""
Prediction Model — The ONE file the AI agent modifies.

Experiment: Add margin-specific interaction features (srs_diff*win_pct_diff,
  elo_diff*margin_diff, srs_diff*elo_diff) and use them only in the margin models.
  These cross-terms should help capture nonlinear relationships in margin prediction
  where the combination of power rating gap and quality gap matters more than either alone.
  Previous MAE: 7.7793
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# Original basic features (always available)
BASIC_FEATURES = [
    "home_ppg", "home_opp_ppg", "home_avg_margin", "home_win_pct",
    "home_last5_ppg", "home_last5_opp_ppg", "home_last5_margin",
    "home_home_ppg", "home_days_rest", "home_games_played",
    "away_ppg", "away_opp_ppg", "away_avg_margin", "away_win_pct",
    "away_last5_ppg", "away_last5_opp_ppg", "away_last5_margin",
    "away_away_ppg", "away_days_rest", "away_games_played",
    "neutral_site",
]

# Enhanced box score features (from CBBData)
ENHANCED_FEATURES = [
    # Efficiency ratings (pace-adjusted)
    "home_avg_off_rating", "home_avg_def_rating",
    "away_avg_off_rating", "away_avg_def_rating",
    # Four Factors (Dean Oliver's keys to winning)
    "home_avg_efg_pct", "home_avg_to_ratio", "home_avg_oreb_pct", "home_avg_ft_rate",
    "away_avg_efg_pct", "away_avg_to_ratio", "away_avg_oreb_pct", "away_avg_ft_rate",
    # Opponent Four Factors (defensive versions)
    "home_avg_opp_efg_pct", "home_avg_opp_to_ratio", "home_avg_opp_oreb_pct", "home_avg_opp_ft_rate",
    "away_avg_opp_efg_pct", "away_avg_opp_to_ratio", "away_avg_opp_oreb_pct", "away_avg_opp_ft_rate",
    # Shooting splits (cumulative, more stable than averaged percentages)
    "home_cum_fg_pct", "home_cum_fg3_pct", "home_cum_ft_pct",
    "away_cum_fg_pct", "away_cum_fg3_pct", "away_cum_ft_pct",
    # Pace and possessions
    "home_avg_pace", "away_avg_pace",
    # Rebounding
    "home_avg_treb", "home_avg_oreb",
    "away_avg_treb", "away_avg_oreb",
    # Ball control
    "home_ast_to_ratio", "away_ast_to_ratio",
    "home_avg_turnovers", "away_avg_turnovers",
    # Play style
    "home_fg3_rate", "away_fg3_rate",
    # Power ratings
    "home_srs", "away_srs",
    "home_elo", "away_elo",
    # Recent form (last 5 games) for key enhanced stats
    "home_last5_off_rating", "home_last5_def_rating",
    "away_last5_off_rating", "away_last5_def_rating",
    "home_last5_efg_pct", "away_last5_efg_pct",
    "home_last5_pace", "away_last5_pace",
]

# Engineered matchup features
ENGINEERED = [
    # Efficiency matchups
    "off_rating_diff",           # home off rating - away off rating
    "def_rating_diff",           # home def rating - away def rating (lower = better D)
    "home_eff_vs_def",           # home off rating vs away def rating
    "away_eff_vs_def",           # away off rating vs home def rating
    # Four Factors differentials
    "efg_diff",                  # home eFG% - away eFG%
    "to_ratio_diff",             # turnover ratio diff (lower = better)
    "oreb_pct_diff",             # offensive rebound % diff
    # Shooting matchup
    "fg3_pct_diff",              # 3-point shooting diff
    # Power rating differentials
    "srs_diff",                  # home SRS - away SRS
    "elo_diff",                  # home Elo - away Elo
    # Pace matchup
    "pace_avg",                  # expected game pace
    "pace_diff",                 # pace style mismatch
    # Original engineered features
    "home_off_vs_def",           # raw ppg matchup
    "away_off_vs_def",
    "margin_diff",
    "recent_margin_diff",
    "rest_diff",
    "win_pct_diff",
    # Trends
    "home_off_trend",
    "away_off_trend",
    "home_def_trend",
    "away_def_trend",
    "home_eff_trend",            # off_rating trend (last5 vs season)
    "away_eff_trend",
]

MARGIN_INTERACTIONS = [
    "srs_x_winpct",
    "elo_x_margin",
    "srs_x_elo",
]

ALL_FEATURES = BASIC_FEATURES + ENHANCED_FEATURES + ENGINEERED
MARGIN_FEATURES = ALL_FEATURES + MARGIN_INTERACTIONS


def _regress_to_mean(stat, games_played, league_avg, k=10):
    """Bayesian shrinkage: blend team stat toward league average based on sample size."""
    weight = games_played / (games_played + k)
    return weight * stat + (1 - weight) * league_avg


def _apply_regression(df):
    """Apply regression to mean for key stats based on games_played."""
    # League averages (approximate D1 averages)
    LEAGUE_AVGS = {
        "ppg": 71.5, "opp_ppg": 71.5, "avg_margin": 0.0,
        "avg_off_rating": 100.0, "avg_def_rating": 100.0,
        "avg_efg_pct": 0.49, "avg_to_ratio": 0.18,
        "avg_pace": 67.5, "srs": 0.0,
    }
    for prefix in ["home_", "away_"]:
        gp_col = f"{prefix}games_played"
        if gp_col not in df.columns:
            continue
        for stat, avg in LEAGUE_AVGS.items():
            col = f"{prefix}{stat}"
            if col in df.columns:
                df[col] = df.apply(
                    lambda row: _regress_to_mean(row[col], row[gp_col], avg, k=8)
                    if pd.notna(row[col]) and pd.notna(row[gp_col])
                    else row[col],
                    axis=1,
                )
    return df


def _add_engineered(df):
    """Add matchup-based engineered features."""
    df = df.copy()

    # Efficiency matchups
    df["off_rating_diff"] = df["home_avg_off_rating"] - df["away_avg_off_rating"]
    df["def_rating_diff"] = df["home_avg_def_rating"] - df["away_avg_def_rating"]
    df["home_eff_vs_def"] = df["home_avg_off_rating"] - df["away_avg_def_rating"]
    df["away_eff_vs_def"] = df["away_avg_off_rating"] - df["home_avg_def_rating"]

    # Four Factors differentials
    df["efg_diff"] = df["home_avg_efg_pct"] - df["away_avg_efg_pct"]
    df["to_ratio_diff"] = df["home_avg_to_ratio"] - df["away_avg_to_ratio"]
    df["oreb_pct_diff"] = df["home_avg_oreb_pct"] - df["away_avg_oreb_pct"]

    # Shooting
    df["fg3_pct_diff"] = df["home_cum_fg3_pct"] - df["away_cum_fg3_pct"]

    # Power ratings
    df["srs_diff"] = df["home_srs"] - df["away_srs"]
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    # Pace
    df["pace_avg"] = (df["home_avg_pace"] + df["away_avg_pace"]) / 2
    df["pace_diff"] = df["home_avg_pace"] - df["away_avg_pace"]

    # Original raw matchups
    df["home_off_vs_def"] = df["home_ppg"] - df["away_opp_ppg"]
    df["away_off_vs_def"] = df["away_ppg"] - df["home_opp_ppg"]
    df["margin_diff"] = df["home_avg_margin"] - df["away_avg_margin"]
    df["recent_margin_diff"] = df["home_last5_margin"] - df["away_last5_margin"]
    df["rest_diff"] = df["home_days_rest"] - df["away_days_rest"]
    df["win_pct_diff"] = df["home_win_pct"] - df["away_win_pct"]

    # Trends
    df["home_off_trend"] = df["home_last5_ppg"] - df["home_ppg"]
    df["away_off_trend"] = df["away_last5_ppg"] - df["away_ppg"]
    df["home_def_trend"] = df["home_opp_ppg"] - df["home_last5_opp_ppg"]
    df["away_def_trend"] = df["away_opp_ppg"] - df["away_last5_opp_ppg"]
    df["home_eff_trend"] = df["home_last5_off_rating"] - df["home_avg_off_rating"]
    df["away_eff_trend"] = df["away_last5_off_rating"] - df["away_avg_off_rating"]

    # Margin-specific interaction features
    df["srs_x_winpct"] = df["srs_diff"] * df["win_pct_diff"]
    df["elo_x_margin"] = df["elo_diff"] * df["margin_diff"]
    df["srs_x_elo"] = df["srs_diff"] * df["elo_diff"]

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
    train = _apply_regression(train_df.copy())
    val = _apply_regression(val_df.copy())

    train = _add_engineered(train)
    val = _add_engineered(val)

    # Compute targets: total and margin
    train["total"] = train["home_score"] + train["away_score"]
    train["margin"] = train["home_score"] - train["away_score"]

    # Filter to features that exist in the data
    available_total = [f for f in ALL_FEATURES if f in train.columns]
    available_margin = [f for f in MARGIN_FEATURES if f in train.columns]

    # Drop rows with missing targets
    train = train.dropna(subset=["total", "margin"])

    X_train_total = train[available_total].astype(float)
    X_val_total = val[available_total].astype(float).fillna(X_train_total.median())
    X_train_total = X_train_total.fillna(X_train_total.median())

    X_train_margin = train[available_margin].astype(float)
    X_val_margin = val[available_margin].astype(float).fillna(X_train_margin.median())
    X_train_margin = X_train_margin.fillna(X_train_margin.median())

    # --- XGBoost predictions ---
    xgb_total = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.06,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=10.0,
        random_state=42, verbosity=0,
    )
    xgb_total.fit(X_train_total, train["total"])
    xgb_pred_total = xgb_total.predict(X_val_total)

    xgb_margin = XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=12.0,
        random_state=42, verbosity=0,
    )
    xgb_margin.fit(X_train_margin, train["margin"])
    xgb_pred_margin = xgb_margin.predict(X_val_margin)

    # --- Ridge regression predictions ---
    scaler_total = StandardScaler()
    X_train_total_sc = scaler_total.fit_transform(X_train_total)
    X_val_total_sc = scaler_total.transform(X_val_total)

    scaler_margin = StandardScaler()
    X_train_margin_sc = scaler_margin.fit_transform(X_train_margin)
    X_val_margin_sc = scaler_margin.transform(X_val_margin)

    ridge_total = Ridge(alpha=50.0)
    ridge_total.fit(X_train_total_sc, train["total"])
    ridge_pred_total = ridge_total.predict(X_val_total_sc)

    ridge_margin = Ridge(alpha=50.0)
    ridge_margin.fit(X_train_margin_sc, train["margin"])
    ridge_pred_margin = ridge_margin.predict(X_val_margin_sc)

    # --- Ensemble: per-target weights ---
    # Total: XGBoost captures nonlinear pace effects better
    # Margin: Ridge regularization helps with noisier target
    pred_total = 0.75 * xgb_pred_total + 0.25 * ridge_pred_total
    pred_margin = 0.55 * xgb_pred_margin + 0.45 * ridge_pred_margin

    # Derive individual scores from total and margin
    val["pred_home_score"] = (pred_total + pred_margin) / 2.0
    val["pred_away_score"] = (pred_total - pred_margin) / 2.0

    return val[["pred_home_score", "pred_away_score"]]
