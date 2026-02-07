from __future__ import annotations
import pandas as pd
from collections import defaultdict
from typing import DefaultDict

K: float = 20.0
BASE_ELO: float = 1500.0


def expected(a: float, b: float) -> float:
    return 1 / (1 + 10**((b - a) / 400))


def build_elo_table(matches_csv: str) -> pd.DataFrame:
    df = pd.read_csv(matches_csv, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    elo: DefaultDict[str, float] = defaultdict(lambda: BASE_ELO)

    rows = []
    for _, r in df.iterrows():
        a = str(r["team_1"])
        b = str(r["team_2"])
        w = str(r["winner"])

        ra = elo[a]
        rb = elo[b]

        ea = expected(ra, rb)
        sa = 1.0 if w == a else 0.0

        rows.append({
            "match_id": str(r["match_id"]),
            "date": r["date"],
            "team_1": a,
            "team_2": b,
            "elo_team_1": ra,
            "elo_team_2": rb,
            "elo_diff": ra - rb
        })

        elo[a] = ra + K * (sa - ea)
        elo[b] = rb + K * ((1 - sa) - (1 - ea))

    return pd.DataFrame(rows)
