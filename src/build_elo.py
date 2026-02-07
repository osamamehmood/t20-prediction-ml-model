from elo import build_elo_table

df = build_elo_table("data/matches_t20i_men.csv")
df.to_csv("data/elo_matches.csv", index=False)
print(df.head())
