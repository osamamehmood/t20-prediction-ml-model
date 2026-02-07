from __future__ import annotations

from datetime import datetime
from itertools import permutations
from math import pow
from src.live_features import build_live_features
from src.predict import predict_proba

import pandas as pd

MATCHES_PATH = "data/matches_t20i_men.csv"
HIST = pd.read_csv(MATCHES_PATH, parse_dates=["date"])

# Tuning knobs (start here)
MIN_MATCHES_START = 10  # below this, rely almost entirely on Elo
MIN_MATCHES_FULL = 50  # at/above this, rely mostly on ML
PROB_CLAMP_LOW = 0.05
PROB_CLAMP_HIGH = 0.95


def elo_expected(elo_a: float, elo_b: float) -> float:
    """Classic Elo expected win probability."""
    return 1 / (1 + pow(10, (elo_b - elo_a) / 400))


def matches_played(team: str, as_of_date: datetime) -> int:
    past = HIST[HIST["date"] < as_of_date]
    tm = past[(past["team_1"] == team) | (past["team_2"] == team)]
    return int(len(tm))


def ml_weight(n_matches: int) -> float:
    """
    Smoothly increase ML influence as history grows.
    0 at MIN_MATCHES_START, 1 at MIN_MATCHES_FULL.
    """
    if n_matches <= MIN_MATCHES_START:
        return 0.0
    if n_matches >= MIN_MATCHES_FULL:
        return 1.0
    return (n_matches - MIN_MATCHES_START) / (MIN_MATCHES_FULL -
                                              MIN_MATCHES_START)


def clamp_prob(p: float) -> float:
    return max(PROB_CLAMP_LOW, min(PROB_CLAMP_HIGH, p))


def build_prob_cache(teams: list[str],
                     as_of_date: datetime) -> dict[tuple[str, str], float]:
    """
    Precompute P(team_a beats team_b) for all ordered pairs in `teams`.

    Strategy:
    - Compute Elo expected win prob p_elo from features
    - Compute ML win prob p_ml from model
    - Blend them based on minimum history between the two teams
      (less history => rely more on Elo)
    """
    cache: dict[tuple[str, str], float] = {}

    # Precompute match counts once for speed
    played = {t: matches_played(t, as_of_date) for t in teams}

    for a, b in permutations(teams, 2):
        feats = build_live_features(a, b, as_of_date)

        # Elo-only baseline probability
        p_elo = elo_expected(float(feats["elo_a"]), float(feats["elo_b"]))

        # ML probability (can be unstable for low-data teams)
        p_ml = float(predict_proba(feats))

        # Weight ML by the weaker-history team in the pairing
        w = ml_weight(min(played[a], played[b]))

        p = w * p_ml + (1.0 - w) * p_elo
        cache[(a, b)] = clamp_prob(p)

    return cache
