from __future__ import annotations

import pandas as pd
from typing import Dict
from datetime import datetime

K_FULL = 20.0
K_ASSOC = 10.0
ASSOCIATE_CAP = 1550.0

FULL_MEMBERS = {
    "India",
    "Australia",
    "England",
    "Pakistan",
    "South Africa",
    "New Zealand",
    "Sri Lanka",
    "West Indies",
    "Afghanistan",
    "Ireland",
    "Zimbabwe",
}


def expected(a: float, b: float) -> float:
    return 1.0 / (1.0 + 10**((b - a) / 400.0))


_matches = (pd.read_csv(
    "data/matches_t20i_men.csv",
    parse_dates=["date"]).sort_values("date").reset_index(drop=True))

for col in ["team_1", "team_2", "winner"]:
    _matches[col] = _matches[col].astype(str).str.strip()

_ELO_CACHE: Dict[str, Dict[str, float]] = {}


def elo_as_of(as_of: datetime) -> Dict[str, float]:
    key = as_of.date().isoformat()
    if key in _ELO_CACHE:
        return _ELO_CACHE[key]

    elo: Dict[str, float] = {}

    def get_rating(team: str) -> float:
        if team not in elo:
            elo[team] = 1550.0 if team in FULL_MEMBERS else 1450.0
        return elo[team]

    past = _matches[_matches["date"] < as_of]

    for r in past.itertuples(index=False):
        a = r.team_1
        b = r.team_2
        w = r.winner

        ra = get_rating(a)
        rb = get_rating(b)

        ea = expected(ra, rb)
        sa = 1.0 if w == a else 0.0

        is_a_full = a in FULL_MEMBERS
        is_b_full = b in FULL_MEMBERS

        k_a = K_FULL if is_a_full else K_ASSOC
        k_b = K_FULL if is_b_full else K_ASSOC

        new_a = ra + k_a * (sa - ea)
        new_b = rb + k_b * ((1.0 - sa) - (1.0 - ea))

        elo[a] = new_a if is_a_full else min(new_a, ASSOCIATE_CAP)
        elo[b] = new_b if is_b_full else min(new_b, ASSOCIATE_CAP)

    _ELO_CACHE[key] = dict(elo)
    return _ELO_CACHE[key]
