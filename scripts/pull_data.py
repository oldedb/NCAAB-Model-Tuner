"""
NCAAB Data Pull Script — ESPN Scoreboard API
Pulls historical game scores from ESPN's public API.

No API key required. Pulls D1 men's basketball game results
including scores, home/away, and neutral site flags.

Output:
  - data/raw_games.csv (one row per game with final scores)
"""

import sys
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# D1 group ID
D1_GROUP = 50


def pull_scoreboard(date_str):
    """Pull all D1 games for a given date (YYYYMMDD)."""
    resp = requests.get(
        f"{ESPN_BASE}/scoreboard",
        params={"dates": date_str, "limit": 200, "groups": D1_GROUP},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def parse_games(scoreboard_data, date_str):
    """Extract completed game records from scoreboard response."""
    games = []
    for event in scoreboard_data.get("events", []):
        comp = event["competitions"][0]

        # Only include completed games
        status = comp.get("status", {}).get("type", {}).get("name", "")
        if status != "STATUS_FINAL":
            continue

        teams = {}
        for competitor in comp["competitors"]:
            side = competitor["homeAway"]
            team = competitor["team"]
            teams[side] = {
                "id": int(team["id"]),
                "name": team.get("displayName", team.get("shortDisplayName", "")),
                "abbrev": team.get("abbreviation", ""),
                "score": int(competitor.get("score", 0)),
            }

        if "home" not in teams or "away" not in teams:
            continue

        games.append({
            "date": date_str,
            "home_id": teams["home"]["id"],
            "home_team": teams["home"]["name"],
            "home_abbrev": teams["home"]["abbrev"],
            "away_id": teams["away"]["id"],
            "away_team": teams["away"]["name"],
            "away_abbrev": teams["away"]["abbrev"],
            "home_score": teams["home"]["score"],
            "away_score": teams["away"]["score"],
            "neutral_site": comp.get("neutralSite", False),
        })

    return games


def pull_season(year):
    """
    Pull all games for one season.
    Year = spring year (2025 means the 2024-25 season).
    Season runs ~Nov 1 through Apr 15.
    """
    start = datetime(year - 1, 11, 1)
    end = datetime(year, 4, 15)
    total_days = (end - start).days

    all_games = []
    current = start
    errors = 0

    while current <= end:
        date_str = current.strftime("%Y%m%d")
        day_num = (current - start).days

        try:
            data = pull_scoreboard(date_str)
            games = parse_games(data, date_str)
            if games:
                all_games.extend(games)
            print(f"\r  Season {year}: day {day_num}/{total_days} | "
                  f"{len(all_games)} games", end="", flush=True)
        except Exception as e:
            errors += 1
            if errors > 10:
                print(f"\n  Too many errors, stopping season {year}")
                break

        current += timedelta(days=1)
        time.sleep(0.25)

    print()
    return all_games


def main():
    DATA_DIR.mkdir(exist_ok=True)

    # Check for cached raw data to support resuming
    cache_path = DATA_DIR / "raw_games.csv"
    if cache_path.exists():
        existing = pd.read_csv(cache_path)
        print(f"Found existing data: {len(existing)} games")
        response = input("Re-pull all data? (y/n): ").strip().lower()
        if response != "y":
            print("Keeping existing data.")
            return

    # Pull seasons: 2021-22 through 2025-26
    # Training: 2022-2024 | Validation: 2025 | Future: 2026
    seasons = [2022, 2023, 2024, 2025, 2026]

    all_games = []
    for year in seasons:
        print(f"\n=== Season {year} ({year-1}-{str(year)[2:]}) ===")
        games = pull_season(year)
        all_games.extend(games)
        print(f"  Total so far: {len(all_games)} games")

        # Save after each season (checkpoint)
        df = pd.DataFrame(all_games)
        df.to_csv(cache_path, index=False)
        print(f"  Checkpoint saved")

    # Final save and summary
    df = pd.DataFrame(all_games)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(cache_path, index=False)

    # Derive season column
    df["season"] = df["date"].apply(
        lambda d: d.year + 1 if d.month >= 10 else d.year
    )

    print(f"\n{'=' * 55}")
    print("SUMMARY")
    print(f"{'=' * 55}")
    print(f"Total games:  {len(df)}")
    print(f"Date range:   {df['date'].min().date()} to {df['date'].max().date()}")
    for season in sorted(df["season"].unique()):
        n = (df["season"] == season).sum()
        print(f"  {season}: {n} games")
    unique_teams = pd.concat([
        df[["home_id", "home_team"]].rename(columns={"home_id": "id", "home_team": "name"}),
        df[["away_id", "away_team"]].rename(columns={"away_id": "id", "away_team": "name"}),
    ]).drop_duplicates("id")
    print(f"Unique teams: {len(unique_teams)}")


if __name__ == "__main__":
    main()
