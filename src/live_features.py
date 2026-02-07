import pandas as pd
from src.live_elo import elo_as_of

MATCHES_PATH = "data/matches_t20i_men.csv"
ROLLING_N = 10
H2H_YEARS = 3

df = pd.read_csv(MATCHES_PATH, parse_dates=["date"])

for col in ["team_1", "team_2", "winner"]:
    df[col] = df[col].astype(str).str.strip()


def team_form(team: str, as_of_date):
    past = df[df["date"] < as_of_date]
    team_matches = past[(past["team_1"] == team) |
                        (past["team_2"] == team)].tail(ROLLING_N)

    # Bayesian smoothing prior
    prior_games = 10
    prior_wins = 5

    if team_matches.empty:
        return prior_wins / prior_games  # 0.5

    wins = (team_matches["winner"] == team).sum()
    games = len(team_matches)

    return (wins + prior_wins) / (games + prior_games)


def head_to_head(team_a: str, team_b: str, as_of_date):
    cutoff = as_of_date - pd.DateOffset(years=H2H_YEARS)
    past = df[(df["date"] < as_of_date)
              & (df["date"] >= cutoff)
              & (((df["team_1"] == team_a) & (df["team_2"] == team_b))
                 | ((df["team_1"] == team_b) & (df["team_2"] == team_a)))]

    # Bayesian smoothing prior
    prior_games = 6
    prior_wins = 3

    if past.empty:
        return prior_wins / prior_games  # 0.5

    wins = (past["winner"] == team_a).sum()
    games = len(past)

    return (wins + prior_wins) / (games + prior_games)


def build_live_features(team_a, team_b, match_date):
    team_a = str(team_a).strip()
    team_b = str(team_b).strip()

    a_form = team_form(team_a, match_date)
    b_form = team_form(team_b, match_date)
    h2h = head_to_head(team_a, team_b, match_date)

    elo = elo_as_of(match_date)
    elo_a = float(elo.get(team_a, 1500.0))
    elo_b = float(elo.get(team_b, 1500.0))

    MIN_ELO = 1450.0
    MAX_ELO = 1950.0
    elo_a = min(max(elo_a, MIN_ELO), MAX_ELO)
    elo_b = min(max(elo_b, MIN_ELO), MAX_ELO)

    ELO_TEMPERATURE = 1.5
    ELO_SCALE = 400.0  # standard Elo scale
    ELO_TEMPERATURE = 1.5  # volatility control

    elo_diff = (elo_a - elo_b) / (ELO_SCALE * ELO_TEMPERATURE)

    # Reduce form impact when Elo gap is large
    elo_gap = abs(elo_a - elo_b)

    FORM_DAMPING_ELO_GAP = 250  # after this, form impact fades

    if elo_gap > FORM_DAMPING_ELO_GAP:
        damp = FORM_DAMPING_ELO_GAP / elo_gap
        a_form = 0.5 + (a_form - 0.5) * damp
        b_form = 0.5 + (b_form - 0.5) * damp

    ELO_FORM_DAMP_START = 75.0
    ELO_FORM_DAMP_FULL = 200.0

    elo_gap = abs(elo_a - elo_b)

    if elo_gap > ELO_FORM_DAMP_START:
        # linear damping factor between 1 → 0
        damp = max(
            0.0, 1.0 - (elo_gap - ELO_FORM_DAMP_START) /
            (ELO_FORM_DAMP_FULL - ELO_FORM_DAMP_START))

        a_form = 0.5 + (a_form - 0.5) * damp
        b_form = 0.5 + (b_form - 0.5) * damp

    FORM_SCALE = 0.4  # 🔑 this is the key

    form_diff = (a_form - b_form) * FORM_SCALE

    return {
        "team_a_form": a_form,
        "team_b_form": b_form,
        "form_diff": form_diff,
        "h2h_win_rate": h2h,
        "elo_a": elo_a,
        "elo_b": elo_b,
        "elo_diff": elo_diff,
    }
