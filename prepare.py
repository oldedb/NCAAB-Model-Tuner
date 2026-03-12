"""
Fixed Evaluation Infrastructure — DO NOT MODIFY.

Loads enhanced game data (box scores, ratings), computes season-to-date
features for each team, splits into train/validation, runs the prediction
model, and evaluates MAE.

Features are computed from per-game box scores — shooting, rebounding,
turnovers, pace, efficiency, plus SRS and Elo power ratings.
All features use only information available BEFORE each game (no leakage).
"""

import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path

ENHANCED_DATA_PATH = Path("data/enhanced_games.csv")
RAW_DATA_PATH = Path("data/raw_games.csv")

# Score sanity bounds
MIN_SCORE = 40
MAX_SCORE = 120


def load_raw_games():
    """Load raw game results (ESPN format, for backward compat)."""
    if not RAW_DATA_PATH.exists():
        print(f"ERROR: {RAW_DATA_PATH} not found. Run scripts/pull_data.py first.")
        sys.exit(1)
    return pd.read_csv(RAW_DATA_PATH, parse_dates=["date"])


def load_enhanced_games():
    """Load enhanced game data with box scores and ratings."""
    if not ENHANCED_DATA_PATH.exists():
        print(f"ERROR: {ENHANCED_DATA_PATH} not found. Run scripts/pull_enhanced_data.py first.")
        sys.exit(1)
    df = pd.read_csv(ENHANCED_DATA_PATH, parse_dates=["date"])
    return df


def add_season(df):
    """Derive season year from game date. Oct-Dec = next year's season."""
    df = df.copy()
    if "season" not in df.columns:
        df["season"] = df["date"].apply(
            lambda d: d.year + 1 if d.month >= 10 else d.year
        )
    return df


def _safe_div(num, denom, default=0):
    """Safe division avoiding divide-by-zero."""
    return num / denom if denom > 0 else default


def compute_team_features(games_df):
    """
    Compute season-to-date features for every team before every game.
    Uses enhanced box score data when available, falls back to basic stats.

    Returns a dict: (team_id, date) -> feature dict.
    """
    games_df = add_season(games_df)
    games_df = games_df.sort_values("date").reset_index(drop=True)

    has_enhanced = "home_fg_pct" in games_df.columns

    # Build flat records: one per team per game
    records = []
    for _, g in games_df.iterrows():
        # Home team record
        home_rec = {
            "date": g["date"], "season": g["season"],
            "team_id": g["home_id"] if "home_id" in g else g.get("home_team_id"),
            "pts_for": g["home_score"], "pts_against": g["away_score"],
            "is_home": True,
        }

        # Away team record
        away_rec = {
            "date": g["date"], "season": g["season"],
            "team_id": g["away_id"] if "away_id" in g else g.get("away_team_id"),
            "pts_for": g["away_score"], "pts_against": g["home_score"],
            "is_home": False,
        }

        if has_enhanced:
            # Enhanced stats for home team
            home_rec.update({
                "fg_pct": g.get("home_fg_pct"),
                "fg3_pct": g.get("home_fg3_pct"),
                "ft_pct": g.get("home_ft_pct"),
                "efg_pct": g.get("home_efg_pct"),
                "oreb": g.get("home_oreb"),
                "dreb": g.get("home_dreb"),
                "treb": g.get("home_treb"),
                "assists": g.get("home_assists"),
                "turnovers": g.get("home_turnovers"),
                "steals": g.get("home_steals"),
                "blocks": g.get("home_blocks"),
                "off_rating": g.get("home_rating"),
                "def_rating": g.get("home_opp_rating"),
                "possessions": g.get("home_possessions"),
                "ft_rate": g.get("home_ft_rate"),
                "to_ratio": g.get("home_to_ratio"),
                "oreb_pct": g.get("home_oreb_pct"),
                "opp_efg_pct": g.get("home_opp_efg_pct"),
                "opp_to_ratio": g.get("home_opp_to_ratio"),
                "opp_oreb_pct": g.get("home_opp_oreb_pct"),
                "opp_ft_rate": g.get("home_opp_ft_rate"),
                "pts_paint": g.get("home_pts_paint"),
                "pts_fastbreak": g.get("home_pts_fastbreak"),
                "pts_off_to": g.get("home_pts_off_to"),
                "fg_made": g.get("home_fg_made"),
                "fg_att": g.get("home_fg_att"),
                "fg3_made": g.get("home_fg3_made"),
                "fg3_att": g.get("home_fg3_att"),
                "ft_made": g.get("home_ft_made"),
                "ft_att": g.get("home_ft_att"),
                "elo": g.get("home_elo_pre"),
                "srs": g.get("home_srs"),
                "pace": g.get("pace"),
            })

            # Enhanced stats for away team
            away_rec.update({
                "fg_pct": g.get("away_fg_pct"),
                "fg3_pct": g.get("away_fg3_pct"),
                "ft_pct": g.get("away_ft_pct"),
                "efg_pct": g.get("away_efg_pct"),
                "oreb": g.get("away_oreb"),
                "dreb": g.get("away_dreb"),
                "treb": g.get("away_treb"),
                "assists": g.get("away_assists"),
                "turnovers": g.get("away_turnovers"),
                "steals": g.get("away_steals"),
                "blocks": g.get("away_blocks"),
                "off_rating": g.get("away_rating"),
                "def_rating": g.get("away_opp_rating"),
                "possessions": g.get("away_possessions"),
                "ft_rate": g.get("away_ft_rate"),
                "to_ratio": g.get("away_to_ratio"),
                "oreb_pct": g.get("away_oreb_pct"),
                "opp_efg_pct": g.get("away_opp_efg_pct"),
                "opp_to_ratio": g.get("away_opp_to_ratio"),
                "opp_oreb_pct": g.get("away_opp_oreb_pct"),
                "opp_ft_rate": g.get("away_opp_ft_rate"),
                "pts_paint": g.get("away_pts_paint"),
                "pts_fastbreak": g.get("away_pts_fastbreak"),
                "pts_off_to": g.get("away_pts_off_to"),
                "fg_made": g.get("away_fg_made"),
                "fg_att": g.get("away_fg_att"),
                "fg3_made": g.get("away_fg3_made"),
                "fg3_att": g.get("away_fg3_att"),
                "ft_made": g.get("away_ft_made"),
                "ft_att": g.get("away_ft_att"),
                "elo": g.get("away_elo_pre"),
                "srs": g.get("away_srs"),
                "pace": g.get("pace"),
            })

        records.append(home_rec)
        records.append(away_rec)

    records_df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

    # Enhanced stat columns to track running averages
    ENHANCED_STATS = [
        "fg_pct", "fg3_pct", "ft_pct", "efg_pct",
        "oreb", "dreb", "treb", "assists", "turnovers", "steals", "blocks",
        "off_rating", "def_rating", "possessions",
        "ft_rate", "to_ratio", "oreb_pct",
        "opp_efg_pct", "opp_to_ratio", "opp_oreb_pct", "opp_ft_rate",
        "pts_paint", "pts_fastbreak", "pts_off_to",
        "fg_made", "fg_att", "fg3_made", "fg3_att", "ft_made", "ft_att",
        "pace",
    ]

    features = {}

    for (team_id, season), group in records_df.groupby(["team_id", "season"]):
        group = group.sort_values("date").reset_index(drop=True)

        pts_for_list = []
        pts_against_list = []
        home_pts = []
        away_pts = []
        dates_played = []
        enhanced_history = {s: [] for s in ENHANCED_STATS}

        for i, row in group.iterrows():
            game_date = row["date"]
            n = len(pts_for_list)

            # Basic features from games BEFORE this one
            if n == 0:
                feat = {
                    "ppg": 0, "opp_ppg": 0, "win_pct": 0, "avg_margin": 0,
                    "games_played": 0, "last5_ppg": 0, "last5_opp_ppg": 0,
                    "last5_margin": 0, "home_ppg": 0, "away_ppg": 0,
                    "days_rest": 7,
                }
            else:
                total_for = sum(pts_for_list)
                total_against = sum(pts_against_list)
                wins = sum(1 for f, a in zip(pts_for_list, pts_against_list) if f > a)
                margins = [f - a for f, a in zip(pts_for_list, pts_against_list)]

                l5_for = pts_for_list[-5:] if n >= 5 else pts_for_list
                l5_against = pts_against_list[-5:] if n >= 5 else pts_against_list
                l5_margins = margins[-5:] if n >= 5 else margins

                days_rest = (game_date - dates_played[-1]).days

                feat = {
                    "ppg": total_for / n,
                    "opp_ppg": total_against / n,
                    "win_pct": wins / n,
                    "avg_margin": sum(margins) / n,
                    "games_played": n,
                    "last5_ppg": sum(l5_for) / len(l5_for),
                    "last5_opp_ppg": sum(l5_against) / len(l5_against),
                    "last5_margin": sum(l5_margins) / len(l5_margins),
                    "home_ppg": sum(home_pts) / len(home_pts) if home_pts else 0,
                    "away_ppg": sum(away_pts) / len(away_pts) if away_pts else 0,
                    "days_rest": min(days_rest, 30),
                }

            # Enhanced features (season-to-date averages from box scores)
            if has_enhanced and n > 0:
                for stat in ENHANCED_STATS:
                    vals = enhanced_history[stat]
                    valid = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
                    if valid:
                        feat[f"avg_{stat}"] = sum(valid) / len(valid)
                        # Last 5 game average
                        recent = valid[-5:] if len(valid) >= 5 else valid
                        feat[f"last5_{stat}"] = sum(recent) / len(recent)
                    else:
                        feat[f"avg_{stat}"] = None
                        feat[f"last5_{stat}"] = None

                # Computed shooting stats from cumulative makes/attempts
                # (more accurate than averaging percentages)
                fg_made_hist = [v for v in enhanced_history["fg_made"] if v is not None and not (isinstance(v, float) and np.isnan(v))]
                fg_att_hist = [v for v in enhanced_history["fg_att"] if v is not None and not (isinstance(v, float) and np.isnan(v))]
                if fg_made_hist and fg_att_hist:
                    feat["cum_fg_pct"] = _safe_div(sum(fg_made_hist), sum(fg_att_hist)) * 100

                fg3_made_hist = [v for v in enhanced_history["fg3_made"] if v is not None and not (isinstance(v, float) and np.isnan(v))]
                fg3_att_hist = [v for v in enhanced_history["fg3_att"] if v is not None and not (isinstance(v, float) and np.isnan(v))]
                if fg3_made_hist and fg3_att_hist:
                    feat["cum_fg3_pct"] = _safe_div(sum(fg3_made_hist), sum(fg3_att_hist)) * 100

                ft_made_hist = [v for v in enhanced_history["ft_made"] if v is not None and not (isinstance(v, float) and np.isnan(v))]
                ft_att_hist = [v for v in enhanced_history["ft_att"] if v is not None and not (isinstance(v, float) and np.isnan(v))]
                if ft_made_hist and ft_att_hist:
                    feat["cum_ft_pct"] = _safe_div(sum(ft_made_hist), sum(ft_att_hist)) * 100

                # Assist-to-turnover ratio
                ast_hist = [v for v in enhanced_history["assists"] if v is not None and not (isinstance(v, float) and np.isnan(v))]
                to_hist = [v for v in enhanced_history["turnovers"] if v is not None and not (isinstance(v, float) and np.isnan(v))]
                if ast_hist and to_hist:
                    feat["ast_to_ratio"] = _safe_div(sum(ast_hist), sum(to_hist))

                # 3-point attempt rate (how much they rely on 3s)
                if fg_att_hist and fg3_att_hist:
                    feat["fg3_rate"] = _safe_div(sum(fg3_att_hist), sum(fg_att_hist)) * 100

            elif has_enhanced and n == 0:
                for stat in ENHANCED_STATS:
                    feat[f"avg_{stat}"] = None
                    feat[f"last5_{stat}"] = None
                feat["cum_fg_pct"] = None
                feat["cum_fg3_pct"] = None
                feat["cum_ft_pct"] = None
                feat["ast_to_ratio"] = None
                feat["fg3_rate"] = None

            # Snapshot Elo and SRS (these are pre-game, not cumulative)
            if has_enhanced:
                feat["elo"] = row.get("elo")
                feat["srs"] = row.get("srs")

            features[(team_id, game_date)] = feat

            # Update running lists
            pts_for_list.append(row["pts_for"])
            pts_against_list.append(row["pts_against"])
            dates_played.append(game_date)
            if row["is_home"]:
                home_pts.append(row["pts_for"])
            else:
                away_pts.append(row["pts_for"])

            # Update enhanced stat history
            if has_enhanced:
                for stat in ENHANCED_STATS:
                    enhanced_history[stat].append(row.get(stat))

    return features


def build_feature_dataset(games_df, team_features):
    """
    Create the final dataset: one row per game with pre-game features for both teams.
    """
    has_home_id = "home_id" in games_df.columns
    rows = []
    for _, g in games_df.iterrows():
        h_id = g["home_id"] if has_home_id else g["home_team_id"]
        a_id = g["away_id"] if has_home_id else g["away_team_id"]

        home_feat = team_features.get((h_id, g["date"]))
        away_feat = team_features.get((a_id, g["date"]))

        if home_feat is None or away_feat is None:
            continue

        if home_feat["games_played"] < 3 or away_feat["games_played"] < 3:
            continue

        row = {
            "date": g["date"],
            "home_id": h_id,
            "home_team": g["home_team"],
            "away_id": a_id,
            "away_team": g["away_team"],
            "home_score": g["home_score"],
            "away_score": g["away_score"],
            "neutral_site": g["neutral_site"],
        }

        for key, val in home_feat.items():
            row[f"home_{key}"] = val
        for key, val in away_feat.items():
            row[f"away_{key}"] = val

        rows.append(row)

    return pd.DataFrame(rows)


def split_data(df):
    """
    Chronological split:
      Train: seasons 2022-2024
      Validation: season 2025
    """
    df = add_season(df)
    train = df[df["season"] <= 2024].copy()
    val = df[df["season"] == 2025].copy()
    return train, val


def sanity_check(predictions):
    """Clamp predicted scores to [40, 120]."""
    predictions = predictions.copy()
    for col in ["pred_home_score", "pred_away_score"]:
        predictions[col] = predictions[col].clip(MIN_SCORE, MAX_SCORE)
    return predictions


def evaluate(predictions, actuals):
    """Calculate MAE and secondary metrics."""
    errors_home = np.abs(predictions["pred_home_score"] - actuals["home_score"])
    errors_away = np.abs(predictions["pred_away_score"] - actuals["away_score"])
    mae = np.mean(np.concatenate([errors_home.values, errors_away.values]))

    pred_margin = predictions["pred_home_score"] - predictions["pred_away_score"]
    actual_margin = actuals["home_score"] - actuals["away_score"]
    margin_mae = np.abs(pred_margin - actual_margin).mean()

    nonzero = actual_margin != 0
    if nonzero.sum() > 0:
        directional_acc = (
            np.sign(pred_margin[nonzero]) == np.sign(actual_margin[nonzero])
        ).mean()
    else:
        directional_acc = 0.0

    pred_total = predictions["pred_home_score"] + predictions["pred_away_score"]
    actual_total = actuals["home_score"] + actuals["away_score"]
    total_mae = np.abs(pred_total - actual_total).mean()

    return {
        "mae": mae,
        "margin_mae": margin_mae,
        "directional_accuracy": directional_acc,
        "total_mae": total_mae,
    }


def main():
    from predict import predict

    # Use enhanced data if available, fall back to raw
    if ENHANCED_DATA_PATH.exists():
        print("Loading enhanced game data...")
        raw = load_enhanced_games()
        # Rename columns to match expected format
        if "home_team_id" in raw.columns and "home_id" not in raw.columns:
            raw = raw.rename(columns={
                "home_team_id": "home_id",
                "away_team_id": "away_id",
            })
        print(f"  {len(raw)} total games (enhanced)")
    else:
        print("Loading raw game data...")
        raw = load_raw_games()
        print(f"  {len(raw)} total games (basic)")

    raw = add_season(raw)

    print("Computing season-to-date features...")
    team_features = compute_team_features(raw)
    print(f"  {len(team_features)} team-game feature records")

    print("Building feature dataset...")
    dataset = build_feature_dataset(raw, team_features)
    dataset = dataset.sort_values("date").reset_index(drop=True)
    print(f"  {len(dataset)} games with features (min 3 games per team)")

    # Show feature count
    feature_cols = [c for c in dataset.columns if c.startswith("home_") and c not in ["home_id", "home_team", "home_score"]]
    print(f"  {len(feature_cols)} features per team ({len(feature_cols) * 2} total)")

    print("Splitting train/validation...")
    train, val = split_data(dataset)
    print(f"  Training:   {len(train)} games (seasons 2022-2024)")
    print(f"  Validation: {len(val)} games (season 2025)")

    if len(val) == 0:
        print("ERROR: No validation data. Check that 2025 season data exists.")
        sys.exit(1)

    print("Running predictions...")
    start = time.time()
    predictions = predict(train, val)
    elapsed = time.time() - start

    if len(predictions) != len(val):
        print(f"ERROR: predict() returned {len(predictions)} rows, expected {len(val)}")
        sys.exit(1)
    for col in ["pred_home_score", "pred_away_score"]:
        if col not in predictions.columns:
            print(f"ERROR: predict() missing column '{col}'")
            sys.exit(1)

    predictions = sanity_check(predictions)
    metrics = evaluate(predictions, val)

    print(f"\n{'=' * 55}")
    print(f"RESULTS  ({elapsed:.1f}s)")
    print(f"{'=' * 55}")
    print(f"  MAE (primary):         {metrics['mae']:.4f} pts")
    print(f"  Margin MAE:            {metrics['margin_mae']:.4f} pts")
    print(f"  Directional Accuracy:  {metrics['directional_accuracy']:.2%}")
    print(f"  Total MAE:             {metrics['total_mae']:.4f} pts")
    print(f"{'=' * 55}")
    print(f"\nSCORE: {metrics['mae']:.4f}")

    return metrics


if __name__ == "__main__":
    main()
