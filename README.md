# 🚀 What this project does

At a high level, the system:

- Uses **historical men’s T20 international data** from Cricsheet
- Builds a dataset of past matches
- Calculates **live Elo ratings** for each team as of a given date
- Engineers match features like:
  - Recent form (rolling window with smoothing)
  - Head-to-head win rate
  - Elo rating difference
- Trains a **machine learning model** to predict match outcomes
- Blends ML predictions with Elo-based probabilities
- Simulates a full tournament (groups → Super 8 → knockouts) using Monte Carlo simulation
- Exposes everything through a simple API so a frontend can consume it

The output you see in the UI (win %, Super 8 %, semi %, final %) is the result of **thousands of simulated tournaments**, not static or hardcoded values.

---

## 🧠 The ML model (simple explanation)

The ML side is intentionally kept simple and explainable:

- **Data source**:  
  Men’s T20I match data from [Cricsheet](https://cricsheet.org/)

- **Core signals used**:
  - **Elo rating**: measures long-term team strength based on historical results
  - **Recent form**: how a team has performed in their last few matches
  - **Head-to-head record**: how two teams have performed against each other
  - **Derived differences** (e.g. Elo difference, form difference)

- **Model**:
  - A lightweight classifier trained on engineered features
  - Outputs a probability that *Team A beats Team B*

- **Live predictions**:
  - Elo is recalculated up to the match date
  - Features are rebuilt on the fly
  - Probabilities change based on date, form, and opponents

Nothing here is magical or overly complex, but it’s realistic, transparent, and easy to improve over time.

---

## 🎲 Tournament simulation

To go beyond single-match predictions, the project simulates full tournaments:

- Group stage (round-robin)
- Super 8 stage
- Semi-finals
- Final

Each match is decided using the predicted probability, plus a small randomness factor.  
Run this **thousands of times**, and you get realistic percentages for:

- Winning the tournament
- Reaching the final
- Reaching the semis
- Qualifying for Super 8

This helps explain *why* a team might look strong overall even if they’re not favourites in every individual match.

---

## 🛠️ Tech stack

- **Python 3.11**
- **FastAPI + Uvicorn** (API)
- **pandas** (data processing)
- **scikit-learn** (machine learning)
- **Cricsheet JSON datasets** (data source)

---
## ▶️ Running the project

### Install dependencies
```bash
uv sync
OR
pip install -r requirements.txt


### Build data
python src/download_cricsheet.py
python src/make_matches_table.py
python src/build_elo.py
python src/build_features.py

### Train the model
python src/train.py

### Run the api
uvicorn api:app --host 0.0.0.0 --port 3000 --reload (locally)
python main.py (prod style)

###API endpoints
GET /health
GET /docs

POST /predict
Example body:
{
  "team_a": "Australia",
  "team_b": "England",
  "date": "2026-02-07"
}

POST /simulate (n_sims = number of simulations)
Example body:
{
  "n_sims": 10000,
}

⚠️ Disclaimer

Of course, this is far from perfect. Cricket is chaotic. Players change, conditions matter, formats evolve, and no model truly knows what’s going to happen. This project is not about being “right”, it’s about learning, experimenting, and having fun.

If you’re a cricket fan and into data or ML, hopefully this sparks some ideas 🙂# t20-prediction-ml-model
