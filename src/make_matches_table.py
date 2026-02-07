from __future__ import annotations

import glob
import json
import os
import pandas as pd

IN_DIR = "data/cricsheet_t20i_json"
OUT_CSV = "data/matches_t20i_men.csv"


def safe_get(d: dict, keys: list[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def parse_match(path: str) -> dict | None:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    info = obj.get("info", {})
    teams = info.get("teams", [])

    # Only keep normal two-team matches
    if not isinstance(teams, list) or len(teams) != 2:
        return None

    outcome = info.get("outcome", {})
    winner = outcome.get("winner")  # can be missing for NR/tie
    result = outcome.get("result")  # e.g. "no result"

    # Dates are usually list, take first
    dates = info.get("dates", [])
    match_date = dates[0] if isinstance(dates, list) and dates else None

    # Toss
    toss = info.get("toss", {})
    toss_winner = toss.get("winner")
    toss_decision = toss.get("decision")

    return {
        "match_id": os.path.splitext(os.path.basename(path))[0],
        "date": match_date,
        "team_1": teams[0],
        "team_2": teams[1],
        "winner": winner,
        "result": result,
        "venue": info.get("venue"),
        "city": info.get("city"),
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "by_runs": safe_get(outcome, ["by", "runs"]),
        "by_wickets": safe_get(outcome, ["by", "wickets"]),
    }


def main() -> None:
    paths = glob.glob(os.path.join(IN_DIR, "**/*.json"), recursive=True)
    if not paths:
        raise SystemExit(f"No JSON files found under {IN_DIR}")

    rows = []
    skipped = 0

    for p in paths:
        try:
            row = parse_match(p)
            if row is None:
                skipped += 1
                continue
            rows.append(row)
        except Exception:
            skipped += 1

    df = pd.DataFrame(rows)

    # Drop matches with no winner for V1 (NR/abandoned/tie). We can add ties later.
    df = df[df["winner"].notna()].copy()

    # Clean dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()
    df["date"] = df["date"].dt.date.astype(str)

    df.to_csv(OUT_CSV, index=False)

    print(f"Parsed files: {len(paths):,}")
    print(f"Skipped: {skipped:,}")
    print(f"Saved rows (with winners only): {len(df):,}")
    print(f"Output: {OUT_CSV}")


if __name__ == "__main__":
    main()
