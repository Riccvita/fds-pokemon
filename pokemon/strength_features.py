"""Label-independent interactions between active HP, stats and stat stages.

These are approximate strength indicators, not a damage simulator: actual move
sets, stat experience, DVs and several battle mechanics are not fully observed.
Only columns already present in the original feature matrix are inspected.
"""

import numpy as np
import pandas as pd


def build_strength_features(frame):
    """Return 42 active-strength interactions, preserving the input row index.

    Approximate non-HP stats use ``2 * base_stat + 100`` before stage modifiers.
    Burn and paralysis modify physical attack and speed respectively. Missing
    species statistics remain missing; zero-HP denominators are bounded below
    by one to avoid infinite pressure indicators. No target or battle ID is used.
    """
    features = {}

    for side in ("p1", "p2"):
        hp = frame[f"{side}_active_hp"]
        for stat in ("atk", "def", "spa", "spd", "spe"):
            stage = frame[f"{side}_boost_{stat}"]
            multiplier = np.where(
                stage >= 0, (2 + stage) / 2,
                2 / np.maximum(2 - stage, 1),
            )
            value = (2 * frame[f"{side}_act_{stat}"] + 100) * multiplier
            if stat == "atk":
                value = value * np.where(frame[f"{side}_active_brn"], 0.5, 1.0)
            elif stat == "spe":
                value = value * np.where(frame[f"{side}_active_par"], 0.25, 1.0)
            features[f"{side}_effective_{stat}"] = value

        features[f"{side}_effective_offense"] = np.maximum(
            features[f"{side}_effective_atk"], features[f"{side}_effective_spa"],
        ) * hp
        for category, stat in (("physical", "def"), ("special", "spd")):
            features[f"{side}_{category}_bulk"] = (
                features[f"{side}_effective_{stat}"]
                * frame[f"{side}_active_maxhp"] * hp
            )

    for side, other in (("p1", "p2"), ("p2", "p1")):
        for attack, defense in (("atk", "def"), ("spa", "spd")):
            ratio = features[f"{side}_effective_{attack}"] / np.maximum(
                features[f"{other}_effective_{defense}"], 1,
            )
            features[f"{side}_matchup_{attack}"] = ratio
            features[f"{side}_pressure_{attack}"] = ratio / np.maximum(
                frame[f"{other}_active_maxhp"] * frame[f"{other}_active_hp"], 1,
            )
        for category, attack in (("physical", "atk"), ("special", "spa")):
            features[f"{side}_{category}_endurance"] = (
                features[f"{side}_{category}_bulk"]
                / np.maximum(features[f"{other}_effective_{attack}"], 1)
            )

    for suffix in [column[3:] for column in features if column.startswith("p1_")]:
        features[f"delta_{suffix}"] = (
            features[f"p1_{suffix}"] - features[f"p2_{suffix}"]
        )

    return pd.DataFrame(features, index=frame.index)
