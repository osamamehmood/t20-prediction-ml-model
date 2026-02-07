# src/features.py
from __future__ import annotations

import pandas as pd
from typing import List, Dict, Any

ROLLING_N = 5
H2H_YEARS = 3
BASE_ELO_FALLBACK = 1500.0


def build_features(matches_csv: str, elo_csv: str) -> pd.DataFrame:
    matches = pd.read_csv(matches_csv)
    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.sort_values("date").reset_index(drop=True)

    elo_df = pd.read_csv(elo_csv)
    elo_df = elo_df[["match_id", "elo_team_1", "elo_team_2", "elo_diff"]]

    df = matches.merge(elo_df, on="match_id", how="left")
    df["date"] = pd.to_datetime(df["date"])

    # Force Elo numeric
    df["elo_team_1"] = pd.to_numeric(df["elo_team_1"], errors="coerce")
    df["elo_team_2"] = pd.to_numeric(df["elo_team_2"], errors="coerce")
    df["elo_diff"] = pd.to_numeric(df["elo_diff"], errors="coerce")

    rows: List[Dict[str, Any]] = []

    # Pre-sort once
    df = df.sort_values("date").reset_index(drop=True)

    for r in df.itertuples(index=False):
        team_a = str(r.team_1)
        team_b = str(r.team_2)
        date = pd.Timestamp(r.date)
        winner = str(r.winner)

        HIST = pd.read_csv("data/matches_t20i_men.csv", parse_dates=["date"])

        past = HIST[HIST["date"] < date]

        def team_form(team: str) -> float:
            team_matches = past[(past["team_1"] == team) |
                                (past["team_2"] == team)].tail(ROLLING_N)
            if team_matches.empty:
                return 0.5
            wins = (team_matches["winner"] == team).sum()
            return float(wins / len(team_matches))

        def head_to_head(a: str, b: str) -> float:
            cutoff = date - pd.DateOffset(years=H2H_YEARS)
            h2h = past[(past["date"] >= cutoff)
                       & (((past["team_1"] == a) & (past["team_2"] == b))
                          | ((past["team_1"] == b) & (past["team_2"] == a)))]
            if h2h.empty:
                return 0.5
            wins = (h2h["winner"] == a).sum()
            return float(wins / len(h2h))

        a_form = team_form(team_a)
        b_form = team_form(team_b)
        h2h_rate = head_to_head(team_a, team_b)

        # Elo values from itertuples are stable and Pyright-friendly
        raw_elo_a = r.elo_team_1
        raw_elo_b = r.elo_team_2
        raw_elo_d = r.elo_diff

        elo_a = float(raw_elo_a) if pd.notna(raw_elo_a) else BASE_ELO_FALLBACK
        elo_b = float(raw_elo_b) if pd.notna(raw_elo_b) else BASE_ELO_FALLBACK
        elo_d = float(raw_elo_d) if pd.notna(raw_elo_d) else (elo_a - elo_b)

        rows.append({
            "date": date,
            "team_a": team_a,
            "team_b": team_b,
            "team_a_form": a_form,
            "team_b_form": b_form,
            "form_diff": a_form - b_form,
            "h2h_win_rate": h2h_rate,
            "elo_a": elo_a,
            "elo_b": elo_b,
            "elo_diff": elo_d,
            "team_a_won": 1 if winner == team_a else 0,
        })

    return pd.DataFrame(rows)
