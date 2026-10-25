"""Switch/KO-estimation features and optional per-turn series.

The battle log under-records P2 knockouts (the fainted state is often skipped
before the replacement appears), so KOs are also estimated from the HP a
Pokemon had when it left the field.
"""

import numpy as np
import pandas as pd


# Numeric codes for status conditions and move categories (used by the per-turn series)
STATUS_CODE = {"nostatus": 0, "slp": 1, "par": 2, "tox": 3, "psn": 4, "frz": 5, "brn": 6, "fnt": 7}
CATEGORY_CODE = {None: 0, "status": 1, "physical": 2, "special": 3}


def feats3(battle, include_series=False):
    """Compute switch / estimated-KO features for one battle.

    With `include_series=True` the per-turn series (HP, status, move category and
    base power for each of the 30 turns, plus the HP advantage) are added too.
    """
    timeline = battle["battle_timeline"]
    f = {}

    for side in ("p1", "p2"):
        state_key = f"{side}_pokemon_state"

        # --- departures: the state of a Pokemon right before it leaves the field
        departures = []
        prev = None

        for turn in timeline:
            st = turn[state_key]
            if prev is not None and prev["name"] != st["name"]:
                departures.append((prev["hp_pct"], prev["status"]))
            prev = st

        for th in (0.1, 0.2, 0.35, 0.5):
            f[f"{side}_dep_lt{th}"] = sum(1 for hp, _ in departures if hp < th)

        f[f"{side}_dep_n"] = len(departures)
        f[f"{side}_dep_hp_mean"] = np.mean([hp for hp, _ in departures]) if departures else 1.0

        # Leaving with high HP most likely means a voluntary switch
        f[f"{side}_dep_hi"] = sum(1 for hp, _ in departures if hp >= 0.5)

        # --- estimated KOs: recorded "fnt" states plus unrecorded departures at very low HP
        recorded = sum(1 for _, status in departures if status == "fnt")
        unrecorded = sum(1 for hp, status in departures if status != "fnt" and hp < 0.12)
        f[f"{side}_ko_est"] = recorded + unrecorded

        # --- optional per-turn series
        if include_series:
            for i, turn in enumerate(timeline):
                st = turn[state_key]
                move = turn[f"{side}_move_details"]

                f[f"{side}_s{i + 1}_hp"] = st["hp_pct"]
                f[f"{side}_s{i + 1}_st"] = STATUS_CODE[st["status"]]
                f[f"{side}_s{i + 1}_cat"] = CATEGORY_CODE[move["category"] if move else None]
                f[f"{side}_s{i + 1}_bp"] = move["base_power"] if move else 0

    if include_series:
        for i in range(1, 31):
            f[f"adv_s{i}"] = f[f"p1_s{i}_hp"] - f[f"p2_s{i}_hp"]

    f["x_ko_est_diff"] = f["p1_ko_est"] - f["p2_ko_est"]

    return f


def build3(battles, include_series=False):
    """Build the switch / estimated-KO feature DataFrame for a list of battles."""
    return pd.DataFrame([feats3(b, include_series) for b in battles])
