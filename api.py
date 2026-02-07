from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import json
from fastapi import Body
from fastapi.middleware.cors import CORSMiddleware

from src.simulate import simulate_tournament
from src.live_features import build_live_features
from src.predict import predict_proba

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5000",
        "https://t20-simulator.replit.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}


class PredictRequest(BaseModel):
    team_a: str
    team_b: str
    date: str  # YYYY-MM-DD


@app.post("/predict")
def predict(req: PredictRequest):
    match_date = datetime.fromisoformat(req.date)

    features = build_live_features(req.team_a, req.team_b, match_date)

    prob_a = predict_proba(features)

    return {
        "team_a": req.team_a,
        "team_b": req.team_b,
        "prob_team_a_win": prob_a,
        "prob_team_b_win": 1 - prob_a,
        "features_used": features,
    }


@app.post("/simulate")
def simulate(payload: dict = Body(...)):
    # payload can contain config directly OR a path
    config = payload.get("config")
    sims = int(payload.get("n_sims", 10000))

    if not config:
        # fallback to local file
        with open("data/t20wc2026_config.json", "r") as f:
            config = json.load(f)

    results = simulate_tournament(config, n_sims=sims)
    return {
        "tournament": config.get("tournament", "T20WC"),
        "n_sims": sims,
        "results": results
    }
