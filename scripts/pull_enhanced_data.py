"""
Pull Enhanced Game Data from College Basketball Data API

Fetches per-game box score stats (shooting, rebounds, turnovers, pace,
four factors, offensive/defensive rating) plus SRS and Elo ratings.

Pulls by conference + season to stay under the 3000-record API cap.
Saves checkpoints after each season.

Usage:
    python scripts/pull_enhanced_data.py
"""

import os
import sys
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# ── Config ──────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("CBBDATA_API_KEY")
BASE_URL = "https://api.collegebasketballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_FILE = DATA_DIR / "enhanced_games.csv"

SEASONS = [2022, 2023, 2024, 2025, 2026]
RATE_LIMIT_PAUSE = 1.0  # seconds between API calls
MAX_RETRIES = 3


def api_get(endpoint, params=None):
    """Make an API request with retries."""
    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                print(f"    Retry {attempt + 1}: {e}")
                time.sleep(5)
            else:
                print(f"    FAILED after {MAX_RETRIES} attempts: {e}")
                return None
    return None


def get_conferences():
    """Get list of all conference abbreviations."""
    data = api_get("conferences")
    if data:
        return [c["abbreviation"] for c in data]
    return []


def get_srs_ratings():
    """Pull SRS ratings for all teams/seasons."""
    print("Pulling SRS ratings...")
    data = api_get("ratings/srs")
    if not data:
        return {}

    # Build lookup: (team, season) -> srs_rating
    srs = {}
    for row in data:
        key = (row["teamId"], row["season"])
        srs[key] = row["rating"]

    print(f"  {len(srs)} team-season SRS ratings loaded")
    return srs


def get_elo_for_season(season, conferences):
    """Pull per-game Elo ratings from games endpoint."""
    print(f"  Pulling Elo ratings for {season}...")
    elo_data = {}  # game_id -> {homeEloStart, homeEloEnd, awayEloStart, awayEloEnd}

    for conf in conferences:
        data = api_get("games", params={"season": season, "conference": conf})
        time.sleep(RATE_LIMIT_PAUSE)
        if not data:
            continue

        for g in data:
            gid = g["id"]
            if gid not in elo_data and g.get("homeTeamEloStart") is not None:
                elo_data[gid] = {
                    "home_elo_pre": g["homeTeamEloStart"],
                    "home_elo_post": g["homeTeamEloEnd"],
                    "away_elo_pre": g["awayTeamEloStart"],
                    "away_elo_post": g["awayTeamEloEnd"],
                }

    print(f"    {len(elo_data)} games with Elo data")
    return elo_data


def extract_team_stats(stats, prefix):
    """Flatten nested teamStats/opponentStats into a flat dict with prefix."""
    if not stats:
        return {}

    row = {}
    # Simple stats
    for key in ["possessions", "assists", "steals", "blocks", "trueShooting", "rating"]:
        row[f"{prefix}_{key}"] = stats.get(key)

    # Game score
    row[f"{prefix}_game_score"] = stats.get("gameScore")

    # Field goals
    for fg_type in ["fieldGoals", "twoPointFieldGoals", "threePointFieldGoals", "freeThrows"]:
        fg = stats.get(fg_type, {}) or {}
        short = {
            "fieldGoals": "fg",
            "twoPointFieldGoals": "fg2",
            "threePointFieldGoals": "fg3",
            "freeThrows": "ft",
        }[fg_type]
        row[f"{prefix}_{short}_made"] = fg.get("made")
        row[f"{prefix}_{short}_att"] = fg.get("attempted")
        row[f"{prefix}_{short}_pct"] = fg.get("pct")

    # Rebounds
    reb = stats.get("rebounds", {}) or {}
    row[f"{prefix}_oreb"] = reb.get("offensive")
    row[f"{prefix}_dreb"] = reb.get("defensive")
    row[f"{prefix}_treb"] = reb.get("total")

    # Turnovers
    to = stats.get("turnovers", {}) or {}
    row[f"{prefix}_turnovers"] = to.get("total")
    row[f"{prefix}_team_turnovers"] = to.get("teamTotal")

    # Fouls
    fouls = stats.get("fouls", {}) or {}
    row[f"{prefix}_fouls"] = fouls.get("total")

    # Points breakdown
    pts = stats.get("points", {}) or {}
    row[f"{prefix}_pts_paint"] = pts.get("inPaint")
    row[f"{prefix}_pts_fastbreak"] = pts.get("fastBreak")
    row[f"{prefix}_pts_off_to"] = pts.get("offTurnovers")
    row[f"{prefix}_largest_lead"] = pts.get("largestLead")

    # Four Factors
    ff = stats.get("fourFactors", {}) or {}
    row[f"{prefix}_efg_pct"] = ff.get("effectiveFieldGoalPct")
    row[f"{prefix}_ft_rate"] = ff.get("freeThrowRate")
    row[f"{prefix}_to_ratio"] = ff.get("turnoverRatio")
    row[f"{prefix}_oreb_pct"] = ff.get("offensiveReboundPct")

    return row


def pull_season_game_stats(season, conferences):
    """Pull per-game box scores for one season via games/teams endpoint."""
    all_records = []
    seen_keys = set()

    for i, conf in enumerate(conferences):
        data = api_get("games/teams", params={"season": season, "conference": conf})
        time.sleep(RATE_LIMIT_PAUSE)

        if not data:
            continue

        # Filter to correct season (API sometimes returns all)
        season_data = [g for g in data if g["season"] == season]

        for g in season_data:
            # Deduplicate: same game can appear via different conference queries
            key = (g["gameId"], g["teamId"])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            all_records.append(g)

        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(conferences)} conferences pulled ({len(all_records)} records)")

    print(f"    Total team-game records: {len(all_records)}")
    return all_records


def pair_home_away(records):
    """Convert team-perspective records into game-level rows (home vs away)."""
    # Group by gameId
    games = {}
    for rec in records:
        gid = rec["gameId"]
        if gid not in games:
            games[gid] = {}

        if rec["isHome"]:
            games[gid]["home"] = rec
        else:
            games[gid]["away"] = rec

    rows = []
    for gid, sides in games.items():
        if "home" not in sides or "away" not in sides:
            continue  # incomplete game (only one side found)

        home = sides["home"]
        away = sides["away"]

        row = {
            "game_id": gid,
            "date": home["startDate"][:10],
            "season": home["season"],
            "season_type": home["seasonType"],
            "home_team_id": home["teamId"],
            "home_team": home["team"],
            "home_conference": home["conference"],
            "away_team_id": away["teamId"],
            "away_team": away["team"],
            "away_conference": away["conference"],
            "neutral_site": home["neutralSite"],
            "conference_game": home["conferenceGame"],
            "pace": home.get("pace"),
        }

        # Home team's stats + what they allowed
        row.update(extract_team_stats(home.get("teamStats"), "home"))
        row.update(extract_team_stats(home.get("opponentStats"), "home_opp"))

        # Away team's stats + what they allowed
        row.update(extract_team_stats(away.get("teamStats"), "away"))
        row.update(extract_team_stats(away.get("opponentStats"), "away_opp"))

        # Points from the points dict
        home_pts = home.get("teamStats", {}).get("points", {}).get("total")
        away_pts = away.get("teamStats", {}).get("points", {}).get("total")
        row["home_score"] = home_pts
        row["away_score"] = away_pts

        rows.append(row)

    return rows


def main():
    if not API_KEY:
        print("ERROR: CBBDATA_API_KEY not found in .env file")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Enhanced Data Pull — College Basketball Data API")
    print("=" * 60)

    # Get conferences
    conferences = get_conferences()
    print(f"Found {len(conferences)} conferences\n")

    # Pull SRS ratings (one call, all seasons)
    srs_ratings = get_srs_ratings()

    # Pull game stats season by season
    all_game_rows = []

    for season in SEASONS:
        print(f"\n{'─' * 40}")
        print(f"Season {season}")
        print(f"{'─' * 40}")

        # Per-game box scores
        records = pull_season_game_stats(season, conferences)
        game_rows = pair_home_away(records)
        print(f"  Paired into {len(game_rows)} games")

        # Elo ratings (not available for 2026)
        elo_data = get_elo_for_season(season, conferences)

        # Merge Elo and SRS into game rows
        for row in game_rows:
            gid = row["game_id"]
            # Elo
            if gid in elo_data:
                row.update(elo_data[gid])
            else:
                row["home_elo_pre"] = None
                row["home_elo_post"] = None
                row["away_elo_pre"] = None
                row["away_elo_post"] = None

            # SRS (season-level)
            row["home_srs"] = srs_ratings.get((row["home_team_id"], season))
            row["away_srs"] = srs_ratings.get((row["away_team_id"], season))

        all_game_rows.extend(game_rows)

        # Checkpoint: save after each season
        df = pd.DataFrame(all_game_rows)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"  Checkpoint saved: {len(all_game_rows)} total games → {OUTPUT_FILE.name}")

    # Final summary
    df = pd.DataFrame(all_game_rows)
    print(f"\n{'=' * 60}")
    print(f"COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total games: {len(df)}")
    print(f"Seasons: {sorted(df['season'].unique())}")
    print(f"Columns: {len(df.columns)}")
    print(f"File: {OUTPUT_FILE}")

    # Show data completeness
    print(f"\nData completeness (non-null %):")
    key_cols = [
        "home_score", "pace", "home_fg_pct", "home_fg3_pct",
        "home_oreb", "home_turnovers", "home_efg_pct", "home_rating",
        "home_elo_pre", "home_srs",
    ]
    for col in key_cols:
        if col in df.columns:
            pct = df[col].notna().mean() * 100
            print(f"  {col:25s} {pct:5.1f}%")

    print(f"\nSample columns:")
    print(f"  {list(df.columns)}")


if __name__ == "__main__":
    main()
