"""Estimate future team resources from observed earlier battle prefixes.

This auxiliary task predicts health-resource changes ten turns into the future,
using only transitions WITHIN the supplied first thirty turns for training.
The final winner label and all unobserved post-turn-30 states are never used.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


FORECAST_HORIZON = 10
TRAIN_CUTOFFS = (5, 10, 15, 20)
FINAL_CUTOFF = 30
STATUS = ("par", "slp", "tox", "psn", "brn", "frz")


@dataclass
class ForecastInputs:
    """Prefix features, observed ten-turn targets, and final-turn states."""

    history: pd.DataFrame
    history_labels: pd.DataFrame
    history_battle: np.ndarray
    final: pd.DataFrame


def _prefix_features(timeline, p1_team, dex, cutoff):
    """Describe the visible state of both rosters at one prefix cutoff."""
    recorded = timeline[:cutoff]
    out = {"forecast_progress": cutoff / 30.0}
    totals = {}
    trajectories = {}

    for side in ("p1", "p2"):
        side_states = [turn[f"{side}_pokemon_state"] for turn in recorded]
        side_moves = [turn[f"{side}_move_details"] for turn in recorded]
        seen_states = {}
        hp_trajectory = []
        statuses = set()
        changes = 0
        damage_moves = 0
        damage_power = 0.0
        previous_species = None

        for state, move in zip(side_states, side_moves):
            seen_states[state["name"]] = state
            if previous_species is not None and previous_species != state["name"]:
                changes += 1
            previous_species = state["name"]
            hp_trajectory.append(
                sum(p["hp_pct"] for p in seen_states.values())
                + max(0, 6 - len(seen_states))
            )
            statuses.add(state["status"])
            if move and move["base_power"] > 0:
                damage_moves += 1
                damage_power += float(move["base_power"])

        active = side_states[-1]
        roster = p1_team if side == "p1" else list(seen_states)
        team_health = hp_trajectory[-1]
        hp = np.asarray(hp_trajectory)
        current = seen_states.values()
        active_stats = dex.get(active["name"], {})

        # --- Only roster composition and conditions known at the cutoff.
        out.update({
            f"{side}_hp_total": team_health,
            f"{side}_hp_seen": sum(s["hp_pct"] for s in current),
            f"{side}_seen_count": len(seen_states),
            f"{side}_fainted": sum(s["hp_pct"] <= 0 for s in current),
            f"{side}_hp_active": active["hp_pct"],
            f"{side}_hp_weak": sum(0 < s["hp_pct"] < 0.35 for s in current),
            f"{side}_swaps": changes,
            f"{side}_swaps_recent": sum(a["name"] != b["name"] for a, b in
                                         zip(side_states[-6:-1], side_states[-5:])),
            f"{side}_status_seen": sum(s["status"] in STATUS for s in current),
            f"{side}_status_active": float(active["status"] in STATUS),
            f"{side}_move_fraction": damage_moves / cutoff,
            f"{side}_power_per_turn": damage_power / cutoff,
            f"{side}_hp_recent5": hp[-5:].mean(),
            f"{side}_hp_last5_delta": hp[-1] - hp[-5],
            f"{side}_hp_prev": hp[0],
            f"{side}_hp_trend": (hp[-1] - hp[0]) / cutoff,
            f"{side}_active_speed": active_stats.get("spe", 75),
            f"{side}_active_atk": active_stats.get("atk", 75),
            f"{side}_active_spa": active_stats.get("spa", 75),
            f"{side}_active_def": active_stats.get("def", 75),
            f"{side}_active_spd": active_stats.get("spd", 75),
            f"{side}_active_boost": sum(active["boosts"].values()),
            f"{side}_speed_alive": sum(dex[name]["spe"] for name in roster
                                         if name in dex and seen_states.get(name, {}).get("hp_pct", 1) > 0),
        })
        for status in STATUS:
            out[f"{side}_active_{status}"] = float(active["status"] == status)

        totals[side] = team_health
        trajectories[side] = hp

    # --- Advantages at snapshot and trends learned from the same visible prefix.
    fields = ("hp_total", "hp_seen", "seen_count", "fainted", "hp_active", "hp_weak",
              "swaps", "status_seen", "move_fraction", "power_per_turn",
              "hp_last5_delta", "hp_trend", "active_speed", "active_boost", "speed_alive")
    for field in fields:
        out[f"diff_{field}"] = out[f"p1_{field}"] - out[f"p2_{field}"]

    return out, totals


def build_forecast_inputs(battles, dex):
    """Produce early-prefix forecast targets and full 30-turn inference features.

    Targets are observed changes in each team's health between t and t+10.
    They are self-supervised and do not depend on ``player_won``.
    """
    history, outcome, history_battle, finals = [], [], [], []

    for index, battle in enumerate(battles):
        timeline = battle["battle_timeline"]
        roster = [p["name"] for p in battle["p1_team_details"]]
        features = {}
        resources = {}

        for cutoff in TRAIN_CUTOFFS + (25, FINAL_CUTOFF):
            features[cutoff], resources[cutoff] = _prefix_features(timeline, roster, dex, cutoff)

        for cutoff in TRAIN_CUTOFFS:
            future = cutoff + FORECAST_HORIZON
            left, right = resources[cutoff], resources[future]
            h1 = right["p1"] - left["p1"]
            h2 = right["p2"] - left["p2"]
            history.append(features[cutoff])
            outcome.append({"p1_health_change": h1, "p2_health_change": h2,
                            "advantage_change": h1 - h2})
            history_battle.append(index)

        finals.append(features[FINAL_CUTOFF])

    final = pd.DataFrame(finals, dtype=np.float32)
    return ForecastInputs(
        history=pd.DataFrame(history, dtype=np.float32).reindex(columns=final.columns),
        history_labels=pd.DataFrame(outcome, dtype=np.float32),
        history_battle=np.asarray(history_battle, dtype=np.int32),
        final=final,
    )


def fit_forecast_models(inputs, train_battles, seed=42):
    """Fit classical regressors on early prefixes of outer training battles."""
    allowed = np.isin(inputs.history_battle, train_battles)
    models = {}

    for name in inputs.history_labels.columns:
        estimator = LGBMRegressor(
            n_estimators=170,
            learning_rate=0.035,
            num_leaves=12,
            max_depth=5,
            min_child_samples=80,
            reg_lambda=12.0,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=2,
            verbosity=-1,
        )
        estimator.fit(inputs.history.loc[allowed], inputs.history_labels.loc[allowed, name])
        models[name] = estimator

    return models


def forecast_features(models, inputs, battle_indices):
    """Predict ten-turn-ahead resource changes at the 30-turn cutoff."""
    frame = inputs.final.iloc[battle_indices].reset_index(drop=True)
    result = pd.DataFrame(index=frame.index)

    for name, estimator in models.items():
        result[f"forecast_{name}"] = estimator.predict(frame)

    hp1 = frame["p1_hp_total"]
    hp2 = frame["p2_hp_total"]
    predicted1 = (hp1 + result["forecast_p1_health_change"]).clip(0, 6)
    predicted2 = (hp2 + result["forecast_p2_health_change"]).clip(0, 6)
    result["forecast_advantage"] = predicted1 - predicted2
    result["forecast_relative_health"] = predicted1 / np.maximum(predicted2, 0.1)
    result["forecast_p1_health"] = predicted1
    result["forecast_p2_health"] = predicted2

    return result.astype(np.float32)
