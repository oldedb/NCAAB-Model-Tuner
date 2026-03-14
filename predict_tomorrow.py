"""
Predict Tomorrow's Games
Fetches tomorrow's NCAAB schedule from ESPN, computes team features
from this season's enhanced box score data, and outputs predicted scores.

Usage:
    python predict_tomorrow.py              # tomorrow's games
    python predict_tomorrow.py 2026-03-15   # specific date
    python predict_tomorrow.py today        # today's games
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

from prepare import load_enhanced_games, load_raw_games, add_season, compute_team_features

load_dotenv()

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
CBBDATA_BASE = "https://api.collegebasketballdata.com"
ENHANCED_DATA_PATH = Path("data/enhanced_games.csv")


def get_schedule(date_str):
    """Fetch games scheduled for a given date from ESPN."""
    resp = requests.get(
        f"{ESPN_BASE}/scoreboard",
        params={"dates": date_str, "limit": 200, "groups": 50},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    games = []
    for event in data.get("events", []):
        comp = event["competitions"][0]

        teams = {}
        for competitor in comp["competitors"]:
            side = competitor["homeAway"]
            team = competitor["team"]
            teams[side] = {
                "id": int(team["id"]),
                "name": team.get("displayName", ""),
                "abbrev": team.get("abbreviation", ""),
            }

        if "home" not in teams or "away" not in teams:
            continue

        games.append({
            "home_espn_id": teams["home"]["id"],
            "home_team": teams["home"]["name"],
            "home_abbrev": teams["home"]["abbrev"],
            "away_espn_id": teams["away"]["id"],
            "away_team": teams["away"]["name"],
            "away_abbrev": teams["away"]["abbrev"],
            "neutral_site": comp.get("neutralSite", False),
        })

    return games


def build_espn_to_cbbdata_map():
    """Build a mapping from ESPN team IDs to CBBData team IDs using the API."""
    api_key = os.getenv("CBBDATA_API_KEY")
    if not api_key:
        return {}

    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = requests.get(f"{CBBDATA_BASE}/teams", headers=headers, timeout=30)
        resp.raise_for_status()
        teams = resp.json()
    except Exception:
        return {}

    # sourceId in CBBData = ESPN's team ID
    mapping = {}
    for t in teams:
        if t.get("sourceId"):
            try:
                mapping[int(t["sourceId"])] = t["id"]
            except (ValueError, TypeError):
                pass

    return mapping


def get_team_current_features(games_df, target_date):
    """
    Compute current season-to-date features for all teams,
    using only games played before target_date.
    """
    games_df = add_season(games_df)
    games_df = games_df.sort_values("date")

    # Only use games before the target date
    games_df = games_df[games_df["date"] < pd.Timestamp(target_date)]

    # Current season
    if target_date.month >= 10:
        current_season = target_date.year + 1
    else:
        current_season = target_date.year

    # Compute features from all data
    team_features = compute_team_features(games_df)

    # Get the most recent feature snapshot for each team in current season
    latest = {}
    for (team_id, game_date), feat in team_features.items():
        game_season = game_date.year + 1 if game_date.month >= 10 else game_date.year
        if game_season == current_season:
            if team_id not in latest or game_date > latest[team_id][0]:
                latest[team_id] = (game_date, feat)

    return {tid: feat for tid, (_, feat) in latest.items()}


def build_prediction_rows(schedule, team_features, target_date, espn_to_cbb):
    """Build a DataFrame matching the format prepare.py produces."""
    rows = []
    skipped = []

    for game in schedule:
        # Map ESPN IDs to CBBData IDs
        home_cbb_id = espn_to_cbb.get(game["home_espn_id"])
        away_cbb_id = espn_to_cbb.get(game["away_espn_id"])

        if home_cbb_id is None or away_cbb_id is None:
            skipped.append(game)
            continue

        home_feat = team_features.get(home_cbb_id)
        away_feat = team_features.get(away_cbb_id)

        if home_feat is None or away_feat is None:
            skipped.append(game)
            continue

        row = {
            "date": pd.Timestamp(target_date),
            "home_id": home_cbb_id,
            "home_team": game["home_team"],
            "away_id": away_cbb_id,
            "away_team": game["away_team"],
            "home_score": 0,
            "away_score": 0,
            "neutral_site": game["neutral_site"],
        }

        for key, val in home_feat.items():
            row[f"home_{key}"] = val
        for key, val in away_feat.items():
            row[f"away_{key}"] = val

        rows.append(row)

    return pd.DataFrame(rows), skipped


def main():
    # Parse date argument
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.lower() == "today":
            target_date = datetime.now()
        else:
            target_date = datetime.strptime(arg, "%Y-%m-%d")
    else:
        target_date = datetime.now() + timedelta(days=1)

    date_str = target_date.strftime("%Y%m%d")
    display_date = target_date.strftime("%A, %B %d, %Y")

    print(f"NCAAB Score Predictions for {display_date}")
    print("=" * 60)

    # 1. Get schedule
    print(f"\nFetching schedule from ESPN...")
    schedule = get_schedule(date_str)

    if not schedule:
        print(f"No games found for {display_date}")
        return

    print(f"Found {len(schedule)} games")

    # 2. Build ESPN -> CBBData team ID mapping
    print("Building team ID mapping...")
    espn_to_cbb = build_espn_to_cbbdata_map()
    print(f"  {len(espn_to_cbb)} teams mapped")

    # 3. Load historical data and compute features
    use_enhanced = ENHANCED_DATA_PATH.exists()
    if use_enhanced:
        print("Loading enhanced game data and computing team stats...")
        raw = load_enhanced_games()
        if "home_team_id" in raw.columns and "home_id" not in raw.columns:
            raw = raw.rename(columns={"home_team_id": "home_id", "away_team_id": "away_id"})
    else:
        print("Loading basic game data and computing team stats...")
        raw = load_raw_games()

    raw["date"] = pd.to_datetime(raw["date"])
    team_features = get_team_current_features(raw, target_date)
    print(f"  Stats available for {len(team_features)} teams")

    # 4. Build prediction input
    pred_df, skipped = build_prediction_rows(schedule, team_features, target_date, espn_to_cbb)

    if pred_df.empty:
        print("Could not build predictions (no team stats found)")
        return

    # 5. Build training data
    from prepare import build_feature_dataset
    all_features = compute_team_features(raw)
    train_df = build_feature_dataset(raw, all_features)
    train_df = add_season(train_df)

    # 6. Run predictions
    from predict import predict
    predictions = predict(train_df, pred_df)

    # 7. Display results
    print(f"\n{'=' * 60}")
    print(f"  PREDICTED SCORES — {display_date}")
    print(f"{'=' * 60}\n")

    for i, (_, game) in enumerate(pred_df.iterrows()):
        home = game["home_team"]
        away = game["away_team"]
        h_score = round(predictions.iloc[i]["pred_home_score"])
        a_score = round(predictions.iloc[i]["pred_away_score"])
        margin = h_score - a_score
        total = h_score + a_score
        neutral = " (N)" if game["neutral_site"] else ""

        winner = home if h_score > a_score else away

        print(f"  {away:<28} {a_score:>3}")
        print(f"  {home:<28} {h_score:>3}{neutral}")
        print(f"  >> {winner} by {abs(margin)} | Total: {total}")
        print()

    if skipped:
        print(f"  ({len(skipped)} games skipped — teams not in database)")

    data_type = "enhanced box scores" if use_enhanced else "basic scores"
    print(f"{'=' * 60}")
    print(f"  Model: XGBoost + Ridge + RF ensemble | Data: {data_type}")
    print(f"  Historical MAE: 7.75 pts")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
