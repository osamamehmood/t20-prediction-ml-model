import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "artifacts/model.pkl"

model = joblib.load(MODEL_PATH)

FEATURES = [
    "team_a_form",
    "team_b_form",
    "form_diff",
    "h2h_win_rate",
    "elo_a",
    "elo_b",
    "elo_diff",
]


def predict_proba(features: dict) -> float:
    """
    Returns probability that Team A wins.
    Blends ML prediction with Elo-only prior.
    """
    # ML-based probability
    df = pd.DataFrame([features])
    ml_p = model.predict_proba(df[FEATURES])[:, 1][0]

    # elo_diff is already scaled (≈ -1 .. +1)
    elo_p = 1.0 / (1.0 + np.exp(-features["elo_diff"]))

    # Blend (Elo-dominant)
    ELO_WEIGHT = 0.65  # 🔑 main control knob (0.6–0.7 is realistic)
    p = ELO_WEIGHT * elo_p + (1.0 - ELO_WEIGHT) * ml_p
    print("predict_proba: ml_p=", round(float(ml_p), 4), "elo_p=",
          round(float(elo_p), 4), "final=", round(float(p), 4))

    return float(min(max(p, 0.01), 0.99))
