"""Base feature engineering: static team data, last-turn battle state and move usage.

All features are computed from the competition data only (train/test jsonl files
and the supplied Generation 2 type chart).
"""

import json
from collections import Counter

import numpy as np
import pandas as pd


# Base stats, in the order used throughout the package
STATS = ["hp", "atk", "def", "spa", "spd", "spe"]

# Stat stages that can be boosted/lowered during a battle (HP excluded)
BOOSTS = ["atk", "def", "spa", "spd", "spe"]

# Major status conditions tracked per active Pokemon
STATUSES = ["slp", "par", "tox", "psn", "frz", "brn"]

# Volatile effects tracked on the active Pokemon
EFFECTS = ["confusion", "substitute", "leechseed", "protect", "perish", "trapped",
           "attract", "encore", "destinybond", "nightmare", "curse"]


def load_type_chart(path):
    """Load the type chart CSV as a dict {(attacking_type, defending_type): multiplier}."""
    chart = pd.read_csv(path)
    return {(a, d): m for a, d, m in chart.itertuples(index=False)}


def type_mult(chart, atk_type, def_types):
    """Damage multiplier of one attacking type against a (possibly dual-typed) defender."""
    mult = 1.0

    for def_type in def_types:
        # "notype" is the placeholder for the missing second type
        if def_type == "notype":
            continue
        mult *= chart.get((atk_type, def_type), 1.0)

    return mult


def read_jsonl(path):
    """Read a jsonl file into a list of dicts (one per battle)."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def side_features(timeline, side):
    """Compute state, trajectory and move features for one side ("p1" or "p2").

    Returns (features, move_counter): the feature dict and a Counter of the
    moves used by this side during the observed turns.
    """
    f = {}
    state_key = f"{side}_pokemon_state"
    move_key = f"{side}_move_details"

    last_hp = {}           # last observed HP fraction per Pokemon
    fainted = set()        # Pokemon seen fainted
    seen = []              # Pokemon in order of first appearance
    status_turns = Counter()
    hp_series = []
    n_switch = 0
    prev_name = None

    # --- single pass over the observed turns
    for turn in timeline:
        st = turn[state_key]
        name = st["name"]

        if name not in seen:
            seen.append(name)

        # A change of active Pokemon counts as a switch (or a forced replacement)
        if prev_name is not None and name != prev_name:
            n_switch += 1
        prev_name = name

        last_hp[name] = st["hp_pct"]

        if st["status"] == "fnt" or st["hp_pct"] == 0:
            fainted.add(name)

        status_turns[st["status"]] += 1
        hp_series.append(st["hp_pct"])

    last = timeline[-1][state_key]

    # --- team-level counters
    f["n_fainted"] = len(fainted)
    f["n_seen"] = len(seen)
    f["n_switch"] = n_switch

    # Estimated remaining team HP: Pokemon never seen are assumed at full health
    f["team_hp_sum"] = sum(last_hp.get(n, 0.0) for n in seen) + (6 - len(seen))
    f["alive_seen_hp"] = sum(v for n, v in last_hp.items() if n not in fainted)

    # --- state of the active Pokemon at the last observed turn
    f["active_hp"] = last["hp_pct"]
    f["active_fnt"] = int(last["status"] == "fnt")

    for s in STATUSES:
        f[f"active_{s}"] = int(last["status"] == s)
        f[f"turns_{s}"] = status_turns[s]

    f["n_status_alive"] = int(last["status"] not in ("nostatus", "fnt"))

    for b in BOOSTS:
        f[f"boost_{b}"] = last["boosts"][b]
    f["boost_sum"] = sum(last["boosts"].values())

    active_effects = set(last["effects"])
    for e in EFFECTS:
        f[f"eff_{e}"] = int(e in active_effects)

    # --- HP trajectory of the active Pokemon
    hp = np.array(hp_series)

    for k in (10, 20, 30):
        f[f"hp_t{k}"] = hp[k - 1]

    f["hp_mean"] = hp.mean()
    f["hp_mean_last10"] = hp[-10:].mean()

    # --- move usage (a missing move means a switch or no action)
    categories = Counter()
    move_counts = Counter()
    bp_sum = 0
    n_moves = 0

    for turn in timeline:
        move = turn[move_key]
        if move is None:
            continue

        n_moves += 1
        categories[move["category"]] += 1
        bp_sum += move["base_power"]
        move_counts[move["name"]] += 1

    f["n_moves"] = n_moves
    f["n_nomove"] = len(timeline) - n_moves

    for c in ("physical", "special", "status"):
        f[f"cat_{c}"] = categories[c]

    f["bp_mean"] = bp_sum / max(n_moves, 1)

    return f, move_counts


def damage_features(timeline, def_side):
    """Estimate the damage taken by `def_side` as the sum of its active-Pokemon HP drops.

    Only turns where the same Pokemon stays active are considered, so that
    switches are not mistaken for damage. Returns (total_drop, n_hits).
    """
    state_key = f"{def_side}_pokemon_state"
    total_drop = 0.0
    n_hits = 0
    prev = None

    for turn in timeline:
        cur = turn[state_key]

        if prev is not None and prev["name"] == cur["name"]:
            drop = prev["hp_pct"] - cur["hp_pct"]
            if drop > 0:
                total_drop += drop
                n_hits += 1

        prev = cur

    return total_drop, n_hits


def build_row(battle, chart):
    """Build the base feature dict for one battle.

    Keys starting with "_" hold raw objects (move counters, name lists) that
    `build_frame` expands into bag-of-words columns.
    """
    timeline = battle["battle_timeline"]
    team = battle["p1_team_details"]
    lead = battle["p2_lead_details"]

    f = {"battle_id": battle["battle_id"]}

    # --- static stats: P1 team aggregates vs the P2 lead
    for s in STATS:
        vals = [p[f"base_{s}"] for p in team]

        f[f"p1_{s}_mean"] = np.mean(vals)
        f[f"p1_{s}_max"] = np.max(vals)
        f[f"p1_{s}_min"] = np.min(vals)
        f[f"lead_{s}"] = lead[f"base_{s}"]
        f[f"d_{s}_mean"] = np.mean(vals) - lead[f"base_{s}"]

    f["p1_bst_mean"] = np.mean([sum(p[f"base_{s}"] for s in STATS) for p in team])
    f["lead_bst"] = sum(lead[f"base_{s}"] for s in STATS)

    # --- type matchup of every P1 Pokemon against the P2 lead (best STAB multiplier)
    off_mult = [max(type_mult(chart, t, lead["types"]) for t in p["types"] if t != "notype")
                for p in team]
    def_mult = [max(type_mult(chart, t, p["types"]) for t in lead["types"] if t != "notype")
                for p in team]

    f["p1_off_mult_mean"] = np.mean(off_mult)
    f["p1_off_mult_max"] = np.max(off_mult)
    f["p1_def_mult_mean"] = np.mean(def_mult)
    f["p1_def_mult_max"] = np.max(def_mult)
    f["p1_n_se"] = sum(m >= 2 for m in off_mult)    # Pokemon super-effective on the lead
    f["p1_n_weak"] = sum(m >= 2 for m in def_mult)  # Pokemon the lead hits super-effectively

    # --- held items of the P1 team
    items = Counter(p["item"] for p in team)
    for item in ("leftovers", "thickclub", "miracleberry", "mintberry",
                 "berserkgene", "goldberry", "berryjuice"):
        f[f"item_{item}"] = items[item]

    # --- per-side state features and their P1 - P2 differences
    f1, moves1 = side_features(timeline, "p1")
    f2, moves2 = side_features(timeline, "p2")

    for k, v in f1.items():
        f[f"p1_{k}"] = v
    for k, v in f2.items():
        f[f"p2_{k}"] = v

    for k in f1:
        f[f"diff_{k}"] = f1[k] - f2[k]

    # --- estimated damage dealt by each side (= HP lost by the opponent)
    dmg1, hits1 = damage_features(timeline, "p2")
    dmg2, hits2 = damage_features(timeline, "p1")

    f["p1_dmg_dealt"], f["p2_dmg_dealt"] = dmg1, dmg2
    f["p1_hits"], f["p2_hits"] = hits1, hits2
    f["diff_dmg"] = dmg1 - dmg2

    # --- evolution of the active-HP advantage over time
    adv = np.array([t["p1_pokemon_state"]["hp_pct"] - t["p2_pokemon_state"]["hp_pct"]
                    for t in timeline])

    f["adv_last"] = adv[-1]
    f["adv_mean"] = adv.mean()
    f["adv_last5"] = adv[-5:].mean()
    f["adv_slope"] = np.polyfit(np.arange(len(adv)), adv, 1)[0]

    # --- raw objects used for bag-of-words expansion
    f["_mv1"] = moves1
    f["_mv2"] = moves2
    f["_team_names"] = [p["name"] for p in team]
    f["_p2_seen"] = list(dict.fromkeys(t["p2_pokemon_state"]["name"] for t in timeline))

    return f


def build_frame(battles, chart, top_moves=None, top_mons=None):
    """Build the base feature DataFrame for a list of battles.

    `top_moves` / `top_mons` define the bag-of-words vocabularies. When None they
    are learned from `battles` (frequency >= 100); pass the training vocabularies
    when encoding the test set so that train and test columns match.

    Returns (DataFrame, top_moves, top_mons).
    """
    records = [build_row(b, chart) for b in battles]

    # Separate the raw objects from the numeric features
    moves1 = [r.pop("_mv1") for r in records]
    moves2 = [r.pop("_mv2") for r in records]
    team_names = [r.pop("_team_names") for r in records]
    p2_seen = [r.pop("_p2_seen") for r in records]

    df = pd.DataFrame(records)

    # --- vocabularies (frequent moves / Pokemon only)
    if top_moves is None:
        counter = Counter()
        for m in moves1 + moves2:
            counter.update(m.keys())
        top_moves = [k for k, v in counter.items() if v >= 100]

    if top_mons is None:
        counter = Counter()
        for names in team_names:
            counter.update(names)
        top_mons = [k for k, v in counter.items() if v >= 100]

    # --- bag-of-moves and bag-of-Pokemon columns
    bag = {}

    for m in top_moves:
        bag[f"p1mv_{m}"] = [x.get(m, 0) for x in moves1]
        bag[f"p2mv_{m}"] = [x.get(m, 0) for x in moves2]

    for mon in top_mons:
        bag[f"p1mon_{mon}"] = [int(mon in names) for names in team_names]
        bag[f"p2seen_{mon}"] = [int(mon in seen) for seen in p2_seen]

    df = pd.concat([df, pd.DataFrame(bag)], axis=1)

    return df, top_moves, top_mons
