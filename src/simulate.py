import random
from datetime import datetime
from itertools import combinations
from src.prob_cache import build_prob_cache
from src.live_features import build_live_features


def margin_bonus(p_winner: float) -> float:
    """
    Proxy for net run rate margin.
    Higher mismatch -> bigger expected margin.
    """
    strength = abs(p_winner - 0.5) * 2.0  # 0..1
    base = 0.08
    scale = 0.25
    noise = random.uniform(-0.03, 0.03)
    return max(0.02, base + scale * strength + noise)


def rank_table(points: dict, nrr: dict):
    return sorted(points.keys(),
                  key=lambda t: (points[t], nrr[t]),
                  reverse=True)


def fixtures_round_robin(teams):
    return list(combinations(teams, 2))


def simulate_match_cached(a, b, prob_cache):
    p_a = prob_cache[(a, b)]
    return a if random.random() < p_a else b


def rank_by_points(points: dict):
    ranked = sorted(points.items(), key=lambda x: x[1], reverse=True)
    return [t for t, _ in ranked]


def simulate_tournament(config: dict, n_sims: int = 10000):
    groups = config["groups"]
    super8_cfg = config["super8"]

    # Build full team list
    all_teams = sorted({t for g in groups.values() for t in g})

    # Counts
    win_counts = {t: 0 for t in all_teams}
    final_counts = {t: 0 for t in all_teams}
    semi_counts = {t: 0 for t in all_teams}
    super8_counts = {
        t: 0
        for t in all_teams
    }  # advanced from group stage into Super 8

    # Use a fixed "as-of" date for features
    as_of_date = datetime(2026, 2, 7)

    # Precompute match probabilities once
    prob_cache = build_prob_cache(all_teams, as_of_date)

    print("DEBUG ELO + probs:")
    pairs = [
        ("South Africa", "Nepal"),
        ("South Africa", "Canada"),
        ("South Africa", "India"),
        ("South Africa", "England"),
    ]

    for a, b in pairs:
        feats = build_live_features(a, b, as_of_date)
        print(
            a,
            "vs",
            b,
            "| p =",
            round(prob_cache[(a, b)], 4),
            "| elo_a =",
            round(float(feats["elo_a"]), 1),
            "| elo_b =",
            round(float(feats["elo_b"]), 1),
            "| elo_diff =",
            round(float(feats["elo_diff"]), 2),
            "| form_a =",
            round(float(feats["team_a_form"]), 2),
            "| form_b =",
            round(float(feats["team_b_form"]), 2),
        )
    print("---")

    for _ in range(n_sims):
        # -----------------------------
        # Group stage
        # -----------------------------
        group_positions = {}  # A -> [1st, 2nd, 3rd, 4th, 5th]

        for group_name, teams in groups.items():
            points = {t: 0 for t in teams}
            nrr = {t: 0.0 for t in teams}

            for a, b in fixtures_round_robin(teams):
                p_a = prob_cache[(a, b)]
                winner = simulate_match_cached(a, b, prob_cache)
                points[winner] += 2

                # NRR proxy update
                if winner == a:
                    m = margin_bonus(p_a)
                    nrr[a] += m
                    nrr[b] -= m
                else:
                    m = margin_bonus(1 - p_a)
                    nrr[b] += m
                    nrr[a] -= m

            group_positions[group_name] = rank_table(points, nrr)

        # Mark Super 8 qualifiers (top 2 from each group)
        qualifiers = []
        for g, ordered in group_positions.items():
            qualifiers.extend(ordered[:2])

        for t in qualifiers:
            super8_counts[t] += 1

        # Helper to resolve "A1", "B2" into real team
        def resolve_group_slot(slot: str) -> str:
            g = slot[0]
            pos = int(slot[1]) - 1
            return group_positions[g][pos]

        # -----------------------------
        # Super 8 stage
        # -----------------------------
        super8_positions = {}  # S1 -> [1st..4th]

        for s_group, slots in super8_cfg["groups"].items():
            s_teams = [resolve_group_slot(s) for s in slots]
            s_points = {t: 0 for t in s_teams}
            s_nrr = {t: 0.0 for t in s_teams}

            for a, b in fixtures_round_robin(s_teams):
                p_a = prob_cache[(a, b)]
                winner = simulate_match_cached(a, b, prob_cache)
                s_points[winner] += 2

                # NRR proxy update (same logic as group stage)
                if winner == a:
                    m = margin_bonus(p_a)
                    s_nrr[a] += m
                    s_nrr[b] -= m
                else:
                    m = margin_bonus(1 - p_a)
                    s_nrr[b] += m
                    s_nrr[a] -= m

            super8_positions[s_group] = rank_table(s_points, s_nrr)

        # Resolve "S1_1" means 1st team in Super 8 group S1
        def resolve_super8_slot(slot: str) -> str:
            s_group, pos = slot.split("_")
            pos_i = int(pos) - 1
            return super8_positions[s_group][pos_i]

        # -----------------------------
        # Knockouts
        # -----------------------------
        sf_pairs = []
        for sf in config["knockout"]["semi_finals"]:
            sf_pairs.append(
                (resolve_super8_slot(sf[0]), resolve_super8_slot(sf[1])))

        sf_winners = []
        for a, b in sf_pairs:
            semi_counts[a] += 1
            semi_counts[b] += 1
            sf_winners.append(simulate_match_cached(a, b, prob_cache))

        final_a, final_b = sf_winners[0], sf_winners[1]
        final_counts[final_a] += 1
        final_counts[final_b] += 1

        champ = simulate_match_cached(final_a, final_b, prob_cache)
        win_counts[champ] += 1

    # Build results
    results = []
    for t in all_teams:
        results.append({
            "team": t,
            "win_pct": win_counts[t] / n_sims * 100,
            "final_pct": final_counts[t] / n_sims * 100,
            "semi_pct": semi_counts[t] / n_sims * 100,
            "super8_pct": super8_counts[t] / n_sims * 100
        })

    results.sort(key=lambda x: x["win_pct"], reverse=True)
    return results
