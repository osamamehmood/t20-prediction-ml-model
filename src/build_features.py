# src/build_features.py
from features import build_features

MATCHES_CSV = "data/matches_t20i_men.csv"
ELO_CSV = "data/elo_matches.csv"
OUTPUT_CSV = "data/features_t20i_men.csv"

df = build_features(MATCHES_CSV, ELO_CSV)
df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved: {OUTPUT_CSV}")
print(f"Rows: {len(df)}")
print(df.head())
