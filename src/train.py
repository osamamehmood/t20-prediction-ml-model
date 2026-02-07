import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib
import json
import os

DATA_PATH = "data/features_t20i_men.csv"
ARTIFACT_DIR = "artifacts"

FEATURES = [
  "team_a_form",
  "team_b_form",
  "form_diff",
  "h2h_win_rate",
  "elo_a",
  "elo_b",
  "elo_diff",
]


TARGET = "team_a_won"


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date")

    X = df[FEATURES]
    y = df[TARGET]

    # Time-based split (last 20% as test)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    # ------------------
    # Baseline model
    # ------------------
    baseline = LogisticRegression(max_iter=1000)
    baseline.fit(X_train, y_train)

    base_probs = baseline.predict_proba(X_test)[:, 1]

    print("\nBaseline Logistic Regression")
    print("Log loss:", log_loss(y_test, base_probs))
    print("Brier score:", brier_score_loss(y_test, base_probs))
    print("Accuracy:", accuracy_score(y_test, base_probs > 0.5))

    # ------------------
    # Gradient Boosting
    # ------------------
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    gb.fit(X_train, y_train)

    gb_probs = gb.predict_proba(X_test)[:, 1]

    print("\nGradient Boosting (uncalibrated)")
    print("Log loss:", log_loss(y_test, gb_probs))
    print("Brier score:", brier_score_loss(y_test, gb_probs))
    print("Accuracy:", accuracy_score(y_test, gb_probs > 0.5))

    # ------------------
    # Calibrate probabilities
    # ------------------
    calibrated = CalibratedClassifierCV(gb, method="isotonic", cv=3)
    calibrated.fit(X_train, y_train)

    cal_probs = calibrated.predict_proba(X_test)[:, 1]

    print("\nGradient Boosting (calibrated)")
    print("Log loss:", log_loss(y_test, cal_probs))
    print("Brier score:", brier_score_loss(y_test, cal_probs))
    print("Accuracy:", accuracy_score(y_test, cal_probs > 0.5))

    # ------------------
    # Save artifacts
    # ------------------
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    joblib.dump(baseline, f"{ARTIFACT_DIR}/model.pkl")

    meta = {
        "features": FEATURES,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "model": "GradientBoostingClassifier + isotonic calibration",
    }

    with open(f"{ARTIFACT_DIR}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nModel saved to artifacts/model.pkl")


if __name__ == "__main__":
    main()
