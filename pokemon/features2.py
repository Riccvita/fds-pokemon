"""Resource-oriented features: absolute HP, alive/fainted strength, KO timing, healing.

Base stats of every Pokemon are collected from the competition data itself
(P1 team members and P2 leads), so no external table is needed.
"""

from collections import Counter

import numpy as np
import pandas as pd

from pokemon.features import STATS


# Moves that restore the user's HP
HEAL_MOVES = ["rest", "recover", "softboiled", "milkdrink", "moonlight", "morningsun", "synthesis"]


def build_dex(*battle_lists):
    """Build a Pokemon "dex" {name: base stats + types} from the given battle lists."""
    dex = {}

    for battles in battle_lists:
        for b in battles:
            for p in b["p1_team_details"] + [b["p2_lead_details"]]:
                dex[p["name"]] = {s: p[f"base_{s}"] for s in STATS} | {"types": p["types"]}

    return dex


def extras(battle, dex):
    """Compute resource features for one battle.

    Returns (features, slots) where `slots` maps "p1hp_<pokemon>" to the last
    observed HP fraction of each P1 team member.
    """
    timeline = battle["battle_timeline"]
    team = battle["p1_team_details"]
    f = {}

    def max_hp(name):
        """Approximate level-100 max HP (max DVs/stat exp); default for unknown names."""
        return 2 * dex[name]["hp"] + 204 if name in dex else 450

    # Average max HP over the dex, used for P2 Pokemon not seen yet
    avg_hp = np.mean([2 * v["hp"] + 204 for v in dex.values()])

    for side in ("p1", "p2"):
        state_key = f"{side}_pokemon_state"

        last_hp = {}
        fainted = set()
        ko_turns = []

        # --- track last HP and KO turns for every Pokemon of this side
        for turn in timeline:
            st = turn[state_key]
            name = st["name"]
            last_hp[name] = st["hp_pct"]

            if (st["status"] == "fnt" or st["hp_pct"] == 0) and name not in fainted:
                fainted.add(name)
                ko_turns.append(turn["turn"])

        seen_names = list(last_hp)

        # P1's full team is known; for P2 only the Pokemon seen so far
        if side == "p1":
            all_names = [p["name"] for p in team]
            n_unseen = sum(1 for p in team if p["name"] not in last_hp)
        else:
            all_names = seen_names
            n_unseen = 6 - len(all_names)

        # --- absolute HP left (unseen Pokemon count as full health)
        abs_hp = sum(last_hp.get(n, 1.0) * max_hp(n) for n in all_names)
        if side == "p2":
            abs_hp += n_unseen * avg_hp

        f[f"{side}_abs_hp"] = abs_hp

        # --- alive Pokemon and their combined strength
        alive = [n for n in all_names if n not in fainted]

        f[f"{side}_alive_n"] = len(alive) + (n_unseen if side == "p2" else 0)

        for s in STATS:
            f[f"{side}_alive_{s}_sum"] = sum(dex[n][s] for n in alive if n in dex)

        f[f"{side}_alive_bst"] = sum(sum(dex[n][s] for s in STATS) for n in alive if n in dex)
        f[f"{side}_fnt_bst"] = sum(sum(dex[n][s] for s in STATS) for n in fainted if n in dex)

        # --- KO timing
        f[f"{side}_ko_first"] = ko_turns[0] if ko_turns else 31
        f[f"{side}_ko_last"] = ko_turns[-1] if ko_turns else 0
        f[f"{side}_ko_recent"] = sum(1 for k in ko_turns if k > 20)

        # --- HP of the active Pokemon over the last three turns
        hp = [t[state_key]["hp_pct"] for t in timeline]
        for i in (1, 2, 3):
            f[f"{side}_hp_last{i}"] = hp[-i]

        # --- healing moves used
        moves = [(t[f"{side}_move_details"] or {}).get("name") for t in timeline]
        f[f"{side}_heal"] = sum(m in HEAL_MOVES for m in moves if m)

        # --- low-HP Pokemon (alive but below 35%)
        f[f"{side}_n_lowhp"] = sum(1 for v in last_hp.values() if 0 < v < 0.35)

        # --- status conditions at each Pokemon's last appearance
        last_status = {}
        for turn in timeline:
            st = turn[state_key]
            last_status[st["name"]] = st["status"]

        f[f"{side}_n_statused"] = sum(1 for s in last_status.values() if s not in ("nostatus", "fnt"))
        f[f"{side}_n_sleep"] = sum(1 for s in last_status.values() if s == "slp")

        # --- base stats of the currently active Pokemon
        active = timeline[-1][state_key]["name"]
        f[f"{side}_active_maxhp"] = max_hp(active)

        for s in STATS:
            f[f"{side}_act_{s}"] = dex[active][s] if active in dex else np.nan

    # --- P1 - P2 differences
    for k in ["abs_hp", "alive_n", "alive_bst", "fnt_bst", "ko_first", "n_lowhp", "n_statused", "heal"]:
        f[f"x_diff_{k}"] = f[f"p1_{k}"] - f[f"p2_{k}"]

    f["x_ratio_abs_hp"] = f["p1_abs_hp"] / max(f["p2_abs_hp"], 1)

    for s in STATS:
        f[f"x_diff_alive_{s}"] = f[f"p1_alive_{s}_sum"] - f[f"p2_alive_{s}_sum"]

    # --- last HP of every P1 team member (unseen = full health)
    last_p1 = {}
    for turn in timeline:
        last_p1[turn["p1_pokemon_state"]["name"]] = turn["p1_pokemon_state"]["hp_pct"]

    slots = {f"p1hp_{p['name']}": last_p1.get(p["name"], 1.0) for p in team}

    # --- number of Spikes layers set by each side
    moves1 = [(t["p1_move_details"] or {}).get("name") for t in timeline]
    moves2 = [(t["p2_move_details"] or {}).get("name") for t in timeline]

    f["p2_spikes_n"] = sum(m == "spikes" for m in moves2)
    f["p1_spikes_n"] = sum(m == "spikes" for m in moves1)

    return f, slots


def build_extras(battles, dex, slot_cols=None):
    """Build the resource-feature DataFrame for a list of battles.

    `slot_cols` is the list of "p1hp_<pokemon>" columns; when None it is learned
    from `battles` (Pokemon present in at least 150 teams). Pass the training
    columns when encoding the test set.

    Returns (DataFrame, slot_cols).
    """
    feature_rows, slot_rows = [], []

    for b in battles:
        feats, slots = extras(b, dex)
        feature_rows.append(feats)
        slot_rows.append(slots)

    df = pd.DataFrame(feature_rows)

    if slot_cols is None:
        counter = Counter()
        for s in slot_rows:
            counter.update(s.keys())
        slot_cols = [k for k, v in counter.items() if v >= 150]

    # Missing slots (Pokemon not in the team) stay NaN, which tree models handle natively
    slot_df = pd.DataFrame([{k: s.get(k, np.nan) for k in slot_cols} for s in slot_rows])

    return pd.concat([df, slot_df], axis=1), slot_cols
