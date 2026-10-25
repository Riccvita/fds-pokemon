"""Pool-based features: inference of unseen P2 Pokemon and matchups between remaining Pokemon.

Teams on both sides are drawn from the same pool of teams, so the Pokemon that P2
has revealed so far can be matched against all known teams (taken from the P1 side
of train and test) to guess which ones are still hidden.
"""

from collections import Counter

import numpy as np

from pokemon.features import STATS, type_mult


# Pokemon whose probability of being among P2's unseen members is exposed as a feature
KEY_MONS = ("snorlax", "zapdos", "raikou", "cloyster", "exeggutor",
            "skarmory", "forretress", "tyranitar", "suicune", "gengar")


def pool_matrix(battles, dex):
    """Build the pool of distinct teams as a boolean membership matrix.

    Returns (keys, M, w, idx, names):
      keys  - list of frozensets of Pokemon names (one per distinct team)
      M     - boolean matrix [n_teams, n_pokemon], True if the team contains the Pokemon
      w     - how many times each team appears in `battles`
      idx   - dict Pokemon name -> column of M
      names - sorted list of Pokemon names (columns of M)
    """
    counts = Counter(frozenset(p["name"] for p in b["p1_team_details"]) for b in battles)

    keys = list(counts)
    names = sorted(dex)
    idx = {n: i for i, n in enumerate(names)}

    M = np.zeros((len(keys), len(names)), bool)
    for i, team in enumerate(keys):
        for n in team:
            M[i, idx[n]] = True

    w = np.array([counts[k] for k in keys], float)

    return keys, M, w, idx, names


def cand_lists(battles, M, idx):
    """For each battle, list the pool teams that contain every P2 Pokemon seen so far.

    A battle where P2 shows a Pokemon outside the pool gets an empty candidate list.
    """
    out = []

    for b in battles:
        seen = {t["p2_pokemon_state"]["name"] for t in b["battle_timeline"]}

        if any(n not in idx for n in seen):
            out.append(np.array([], int))
            continue

        cols = [idx[n] for n in seen]
        out.append(np.where(M[:, cols].all(axis=1))[0])

    return out


def feats4(battle, cands, M, w, names, dex, chart):
    """Compute pool-inference and matchup features for one battle.

    `cands` are the indices of the pool teams compatible with P2's seen Pokemon.
    """
    timeline = battle["battle_timeline"]
    f = {}

    # --- P2 Pokemon seen so far, in order of appearance
    seen = []
    for t in timeline:
        n = t["p2_pokemon_state"]["name"]
        if n not in seen:
            seen.append(n)

    name_idx = {n: i for i, n in enumerate(names)}
    n_unseen = 6 - len(seen)

    f["cand_n"] = len(cands)
    f["cand_w"] = w[cands].sum() if len(cands) else 0

    # Per-Pokemon strength vectors (aligned with the columns of M)
    bst = np.array([sum(dex[n][s] for s in STATS) for n in names], float)
    max_hp = np.array([2 * dex[n]["hp"] + 204 for n in names], float)
    seen_cols = [name_idx[n] for n in seen if n in name_idx]

    # --- expected strength of the unseen P2 Pokemon, averaged over candidate teams
    if len(cands) and n_unseen > 0:
        weights = w[cands] / w[cands].sum()

        # Keep only the members that are not already seen
        hidden = M[cands].copy()
        hidden[:, seen_cols] = False

        f["p2_unseen_bst_exp"] = float((hidden * bst).sum(1) @ weights)
        f["p2_unseen_hp_exp"] = float((hidden * max_hp).sum(1) @ weights)

        # Probability that each Pokemon is among the hidden ones
        prob = (hidden * weights[:, None]).sum(0)
        f["p2_unseen_maxprob"] = float(prob.max())

        for n in KEY_MONS:
            f[f"p2_unseen_p_{n}"] = float(prob[name_idx[n]]) if n in name_idx else 0.0
    else:
        # Nothing hidden (6 seen) -> 0; no usable candidate -> unknown (NaN)
        f["p2_unseen_bst_exp"] = 0.0 if n_unseen == 0 else np.nan
        f["p2_unseen_hp_exp"] = 0.0 if n_unseen == 0 else np.nan
        f["p2_unseen_maxprob"] = 0.0

        for n in KEY_MONS:
            f[f"p2_unseen_p_{n}"] = 0.0

    # --- last HP and fainted Pokemon of both sides
    last1, last2 = {}, {}
    fainted1, fainted2 = set(), set()

    for t in timeline:
        for side, last, fainted in (("p1", last1, fainted1), ("p2", last2, fainted2)):
            st = t[f"{side}_pokemon_state"]
            last[st["name"]] = st["hp_pct"]

            if st["status"] == "fnt" or st["hp_pct"] == 0:
                fainted.add(st["name"])

    team = [p["name"] for p in battle["p1_team_details"]]
    alive1 = [n for n in team if n not in fainted1]
    alive2 = [n for n in seen if n not in fainted2 and n in dex]

    def best_stab(attacker, defender):
        """Best STAB multiplier of `attacker`'s types against `defender`."""
        return max(type_mult(chart, t, dex[defender]["types"])
                   for t in dex[attacker]["types"] if t != "notype")

    # --- matchups between the remaining Pokemon of the two sides
    if alive1 and alive2:
        offense = np.array([[best_stab(a, b) for b in alive2] for a in alive1])
        defense = np.array([[best_stab(b, a) for b in alive2] for a in alive1])

        f["mx_off_mean"] = offense.mean()
        f["mx_def_mean"] = defense.mean()
        f["mx_net"] = offense.mean() - defense.mean()
        f["mx_off_best_mean"] = offense.max(0).mean()
        f["mx_def_worst"] = defense.max(0).mean()

        # Fraction of (P1, P2) pairs where the P1 Pokemon is faster
        speed1 = np.array([dex[a]["spe"] for a in alive1])
        speed2 = np.array([dex[b]["spe"] for b in alive2])
        f["spe_faster_frac"] = np.mean(speed1[:, None] > speed2[None, :])

        # Remaining offensive/defensive power weighted by remaining HP
        for s in ("atk", "spa", "def", "spd"):
            f[f"wt_{s}_diff"] = (sum(dex[a][s] * last1.get(a, 1.0) for a in alive1)
                                 - sum(dex[b][s] * last2.get(b, 1.0) for b in alive2))
    else:
        for k in ["mx_off_mean", "mx_def_mean", "mx_net", "mx_off_best_mean", "mx_def_worst",
                  "spe_faster_frac", "wt_atk_diff", "wt_spa_diff", "wt_def_diff", "wt_spd_diff"]:
            f[k] = np.nan

    f["x_alive_a1"] = len(alive1)
    f["x_alive_a2seen"] = len(alive2)

    return f
