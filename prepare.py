"""
Fixed Evaluation Infrastructure — DO NOT MODIFY.

Loads raw game data, computes season-to-date features for each team,
splits into train/validation, runs the prediction model, and evaluates MAE.

Features are computed from game results only — no external ratings.
All features use only information available BEFORE each game (no leakage).
"""

import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DATA_PATH = Path("data/raw_games.csv")

# Score sanity bounds
MIN_SCORE = 40
MAX_SCORE = 120


def load_raw_games():
    """Load raw game results."""
    if not RAW_DATA_PATH.exists():
        print(f"ERROR: {RAW_DATA_PATH} not found. Run scripts/pull_data.py first.")
        sys.exit(1)
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["date"])
    return df


def add_season(df):
    """Derive season year from game date. Oct-Dec = next year's season."""
    df = df.copy()
    df["season"] = df["date"].apply(
        lambda d: d.year + 1 if d.month >= 10 else d.year
    )
    return df


def compute_team_features(games_df):
    """
    Compute season-to-date features for every team before every game.
    Returns a dict: (team_id, season, date) -> feature dict.

    Features computed:
      - ppg: points per game (season to date)
      - opp_ppg: opponent points per game (defensive)
      - win_pct: winning percentage
      - avg_margin: average scoring margin
      - games_played: number of games played so far
      - last5_ppg: PPG over last 5 games
      - last5_opp_ppg: opponent PPG over last 5 games
      - last5_margin: avg margin over last 5 games
      - home_ppg: PPG in home games
      - away_ppg: PPG in away games
      - days_rest: days since last game
    """
    games_df = add_season(games_df)
    games_df = games_df.sort_values("date").reset_index(drop=True)

    # Build a flat list of (date, season, team_id, pts_for, pts_against, is_home)
    records = []
    for _, g in games_df.iterrows():
        records.append({
            "date": g["date"], "season": g["season"],
            "team_id": g["home_id"], "pts_for": g["home_score"],
            "pts_against": g["away_score"], "is_home": True,
        })
        records.append({
            "date": g["date"], "season": g["season"],
            "team_id": g["away_id"], "pts_for": g["away_score"],
            "pts_against": g["home_score"], "is_home": False,
        })

    records_df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

    # For each team+season, compute cumulative stats BEFORE each game
    features = {}  # (team_id, date) -> feature dict

    for (team_id, season), group in records_df.groupby(["team_id", "season"]):
        group = group.sort_values("date").reset_index(drop=True)

        pts_for_list = []
        pts_against_list = []
        home_pts = []
        away_pts = []
        dates_played = []

        for i, row in group.iterrows():
            game_date = row["date"]
            n = len(pts_for_list)

            # Features from games BEFORE this one
            if n == 0:
                feat = {
                    "ppg": 0, "opp_ppg": 0, "win_pct": 0, "avg_margin": 0,
                    "games_played": 0, "last5_ppg": 0, "last5_opp_ppg": 0,
                    "last5_margin": 0, "home_ppg": 0, "away_ppg": 0,
                    "days_rest": 7,  # default for first game
                }
            else:
                total_for = sum(pts_for_list)
                total_against = sum(pts_against_list)
                wins = sum(1 for f, a in zip(pts_for_list, pts_against_list) if f > a)
                margins = [f - a for f, a in zip(pts_for_list, pts_against_list)]

                # Last 5 games
                l5_for = pts_for_list[-5:] if n >= 5 else pts_for_list
                l5_against = pts_against_list[-5:] if n >= 5 else pts_against_list
                l5_margins = margins[-5:] if n >= 5 else margins

                # Days rest
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

            features[(team_id, game_date)] = feat

            # Update running lists
            pts_for_list.append(row["pts_for"])
            pts_against_list.append(row["pts_against"])
            dates_played.append(game_date)
            if row["is_home"]:
                home_pts.append(row["pts_for"])
            else:
                away_pts.append(row["pts_for"])

    return features


def build_feature_dataset(games_df, team_features):
    """
    Create the final dataset: one row per game with pre-game features for both teams.
    """
    rows = []
    for _, g in games_df.iterrows():
        home_feat = team_features.get((g["home_id"], g["date"]))
        away_feat = team_features.get((g["away_id"], g["date"]))

        if home_feat is None or away_feat is None:
            continue

        # Skip games where either team has played fewer than 3 games
        if home_feat["games_played"] < 3 or away_feat["games_played"] < 3:
            continue

        row = {
            "date": g["date"],
            "home_id": g["home_id"],
            "home_team": g["home_team"],
            "away_id": g["away_id"],
            "away_team": g["away_team"],
            "home_score": g["home_score"],
            "away_score": g["away_score"],
            "neutral_site": g["neutral_site"],
        }

        # Add prefixed features for both teams
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
    """
    Calculate MAE and secondary metrics.
    """
    errors_home = np.abs(predictions["pred_home_score"] - actuals["home_score"])
    errors_away = np.abs(predictions["pred_away_score"] - actuals["away_score"])
    mae = np.mean(np.concatenate([errors_home.values, errors_away.values]))

    # Margin analysis
    pred_margin = predictions["pred_home_score"] - predictions["pred_away_score"]
    actual_margin = actuals["home_score"] - actuals["away_score"]
    margin_mae = np.abs(pred_margin - actual_margin).mean()

    # Directional accuracy
    nonzero = actual_margin != 0
    if nonzero.sum() > 0:
        directional_acc = (
            np.sign(pred_margin[nonzero]) == np.sign(actual_margin[nonzero])
        ).mean()
    else:
        directional_acc = 0.0

    # Total accuracy
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

    print("Loading raw game data...")
    raw = load_raw_games()
    raw = add_season(raw)
    print(f"  {len(raw)} total games")

    print("Computing season-to-date features...")
    team_features = compute_team_features(raw)
    print(f"  {len(team_features)} team-game feature records")

    print("Building feature dataset...")
    dataset = build_feature_dataset(raw, team_features)
    dataset = dataset.sort_values("date").reset_index(drop=True)
    print(f"  {len(dataset)} games with features (min 3 games per team)")

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

    # Validate output
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
