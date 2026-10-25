"""Model observable next-turn decisions with classical machine learning.

A turn's action is classified as damage, status or no recorded move. The
third class may include voluntary switches, forced replacements or an action
that was not recorded; it must not be interpreted as a clean switch label.

Training labels come *only* from moves in the 30 observed turns. Features at
cutoff t use turns <= t; a training label comes from turn t+1. For inference,
the cutoff is turn 30, and only the observed prefix is available. Winner labels
and post-turn-30 battle outcomes are never used to fit this action model.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from pokemon.features import type_mult


CUTOFFS = (9, 14, 19, 24, 29)
CLASSES = ("unrecorded", "status", "damage")


@dataclass
class ActionInputs:
    """Data for fitting and applying the auxiliary action classifier.

    The history rows are ordered by battle, cutoff and player. ``battle_index``
    maps each target to the source battle and permits group-safe outer folds.
    ``final`` has precisely two rows per battle, ordered P1 then P2.
    """

    history: pd.DataFrame
    history_target: np.ndarray
    history_battle_index: np.ndarray
    final: pd.DataFrame


def _action_code(move):
    """Encode a recorded move as one of three broad decision classes."""
    if move is None:
        return 0
    return 1 if move["category"] == "status" else 2


def _matchup_strength(move, attacker_name, target_name, dex, chart):
    """Approximate offensive potential from supplied moves, stats and types."""
    attacker = dex.get(attacker_name)
    defender = dex.get(target_name)

    if not attacker or not defender or not move or move["base_power"] <= 0:
        return 0.0

    physical = move["category"] == "physical"
    attack_stat = "atk" if physical else "spa"
    defense_stat = "def" if physical else "spd"
    stab = 1.5 if move["type"] in attacker["types"] else 1.0
    effectiveness = type_mult(chart, move["type"], defender["types"])

    return (float(move["base_power"]) * float(move["accuracy"]) * stab
            * effectiveness * attacker[attack_stat] / max(defender[defense_stat], 1))


def _snapshot(timeline, cutoff, side, dex, chart):
    """Describe information known immediately after a cutoff turn.

    Snapshot extraction uses no data from later turns and keeps the same
    feature schema for both players. Historical action frequencies and the
    revealed moves of the current species summarize the player's strategy.
    """
    own = f"p{side}"
    other = f"p{3 - side}"
    observed = timeline[:cutoff]

    own_states = [turn[f"{own}_pokemon_state"] for turn in observed]
    other_states = [turn[f"{other}_pokemon_state"] for turn in observed]
    own_moves = [turn[f"{own}_move_details"] for turn in observed]
    other_moves = [turn[f"{other}_move_details"] for turn in observed]

    active = own_states[-1]
    enemy = other_states[-1]

    # --- Construct a causal summary of each team's last observed resources.
    own_last = {state["name"]: state for state in own_states}
    other_last = {state["name"]: state for state in other_states}

    def resources(last):
        """Count visible remaining team health, with hidden members at full HP."""
        total = sum(state["hp_pct"] for state in last.values())
        total += max(0, 6 - len(last))
        healthy = sum(state["hp_pct"] > 0 for state in last.values())
        healthy += max(0, 6 - len(last))
        return total, healthy

    own_total, own_alive = resources(own_last)
    other_total, other_alive = resources(other_last)

    # --- Restrict the action history and revealed moves to the cutoff prefix.
    act = np.asarray([_action_code(m) for m in own_moves], dtype=np.int8)
    opp_act = np.asarray([_action_code(m) for m in other_moves], dtype=np.int8)
    current_species_actions = [
        _action_code(move)
        for state, move in zip(own_states, own_moves)
        if state["name"] == active["name"]
    ]

    revealed = {}
    for state, move in zip(own_states, own_moves):
        if state["name"] == active["name"] and move is not None:
            revealed[move["name"]] = move

    damage_power = sorted((
        _matchup_strength(move, active["name"], enemy["name"], dex, chart)
        for move in revealed.values()
    ), reverse=True)

    other_revealed = {}
    for state, move in zip(other_states, other_moves):
        if state["name"] == enemy["name"] and move is not None:
            other_revealed[move["name"]] = move

    enemy_power = sorted((
        _matchup_strength(move, enemy["name"], active["name"], dex, chart)
        for move in other_revealed.values()
    ), reverse=True)

    own_data = dex.get(active["name"], {})
    enemy_data = dex.get(enemy["name"], {})

    # --- Aggregate action choices on multiple horizons and active species.
    row = {
        "turn_progress": cutoff / 30.0,
        "player_is_p1": float(side == 1),
        "own_hp": active["hp_pct"],
        "enemy_hp": enemy["hp_pct"],
        "hp_advantage": active["hp_pct"] - enemy["hp_pct"],
        "own_team_hp": own_total,
        "enemy_team_hp": other_total,
        "team_hp_advantage": own_total - other_total,
        "own_alive": own_alive,
        "enemy_alive": other_alive,
        "own_seen": len(own_last),
        "enemy_seen": len(other_last),
        "own_known_fainted": sum(s["hp_pct"] <= 0 for s in own_last.values()),
        "enemy_known_fainted": sum(s["hp_pct"] <= 0 for s in other_last.values()),
        "own_speed": own_data.get("spe", 75),
        "enemy_speed": enemy_data.get("spe", 75),
        "speed_ratio": own_data.get("spe", 75) / max(enemy_data.get("spe", 75), 1),
        "own_attack": own_data.get("atk", 75),
        "own_special": own_data.get("spa", 75),
        "enemy_defense": enemy_data.get("def", 75),
        "enemy_special_defense": enemy_data.get("spd", 75),
        "own_status": float(active["status"] != "nostatus"),
        "enemy_status": float(enemy["status"] != "nostatus"),
        "own_sleep": float(active["status"] == "slp"),
        "own_paralysis": float(active["status"] == "par"),
        "own_poison": float(active["status"] in ("tox", "psn")),
        "enemy_sleep": float(enemy["status"] == "slp"),
        "own_boost_total": sum(active["boosts"].values()),
        "enemy_boost_total": sum(enemy["boosts"].values()),
        "own_speed_boost": active["boosts"]["spe"],
        "own_attack_boost": active["boosts"]["atk"],
        "own_special_boost": active["boosts"]["spa"],
        "own_has_substitute": float("substitute" in active["effects"]),
        "enemy_has_substitute": float("substitute" in enemy["effects"]),
        "own_active_duration": next((j for j, state in enumerate(reversed(own_states))
                                     if state["name"] != active["name"]), cutoff),
        "enemy_active_duration": next((j for j, state in enumerate(reversed(other_states))
                                       if state["name"] != enemy["name"]), cutoff),
        "own_best_revealed_power": damage_power[0] if damage_power else 0.0,
        "own_mean_revealed_power": np.mean(damage_power) if damage_power else 0.0,
        "enemy_best_revealed_power": enemy_power[0] if enemy_power else 0.0,
        "own_revealed_move_count": len(revealed),
        "enemy_revealed_move_count": len(other_revealed),
        "own_hp_change_last5": own_states[-1]["hp_pct"] - own_states[-min(5, cutoff)]["hp_pct"]
        if own_states[-1]["name"] == own_states[-min(5, cutoff)]["name"] else 0.0,
        "enemy_hp_change_last5": other_states[-1]["hp_pct"] - other_states[-min(5, cutoff)]["hp_pct"]
        if other_states[-1]["name"] == other_states[-min(5, cutoff)]["name"] else 0.0,
        "last_action": int(act[-1]),
        "last_enemy_action": int(opp_act[-1]),
        "active_species_n_actions": len(current_species_actions),
        "own_switch_count": sum(a["name"] != b["name"] for a, b in
                                zip(own_states[:-1], own_states[1:])),
        "enemy_switch_count": sum(a["name"] != b["name"] for a, b in
                                  zip(other_states[:-1], other_states[1:])),
    }

    for window in (5, 10, 20):
        own_recent = act[-window:]
        enemy_recent = opp_act[-window:]
        row[f"own_damage_rate_{window}"] = np.mean(own_recent == 2)
        row[f"own_status_rate_{window}"] = np.mean(own_recent == 1)
        row[f"own_unrecorded_rate_{window}"] = np.mean(own_recent == 0)
        row[f"enemy_damage_rate_{window}"] = np.mean(enemy_recent == 2)
        row[f"enemy_status_rate_{window}"] = np.mean(enemy_recent == 1)
        row[f"enemy_unrecorded_rate_{window}"] = np.mean(enemy_recent == 0)

    for action in range(3):
        row[f"current_species_action_rate_{action}"] = (
            np.mean(np.asarray(current_species_actions) == action)
            if current_species_actions else 0.0
        )

    return row


def build_action_inputs(battles, dex, chart):
    """Collect next-action targets and 30-turn prediction contexts.

    Returns
    -------
    ActionInputs
        Every action learner target is sourced from turns 6..30. The
        recorded turn-30 state is never paired with an unobserved label.
    """
    history, target, battle_idx, final = [], [], [], []

    for index, battle in enumerate(battles):
        timeline = battle["battle_timeline"]
        for cutoff in CUTOFFS:
            if cutoff >= len(timeline):
                continue
            for side in (1, 2):
                snapshot = _snapshot(timeline, cutoff, side, dex, chart)
                next_move = timeline[cutoff][f"p{side}_move_details"]
                history.append(snapshot)
                target.append(_action_code(next_move))
                battle_idx.append(index)

        for side in (1, 2):
            final.append(_snapshot(timeline, len(timeline), side, dex, chart))

    final_frame = pd.DataFrame(final, dtype=np.float32)
    history_frame = pd.DataFrame(history, dtype=np.float32)

    return ActionInputs(
        history=history_frame.reindex(columns=final_frame.columns),
        history_target=np.asarray(target, dtype=np.int8),
        history_battle_index=np.asarray(battle_idx, dtype=np.int32),
        final=final_frame,
    )


def fit_action_model(inputs, train_battles, seed=42):
    """Fit next-action probabilities using outer-training battles only."""
    mask = np.isin(inputs.history_battle_index, train_battles)
    estimator = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=180,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=75,
        colsample_bytree=0.8,
        reg_lambda=10.0,
        random_state=seed,
        n_jobs=3,
        verbosity=-1,
    )
    estimator.fit(inputs.history.loc[mask], inputs.history_target[mask])
    return estimator


def action_features(model, inputs, battle_indices):
    """Build outcome-model features from predicted turn-31 action choices.

    Action probabilities are multiplied by offensive matchup proxies to make
    the predicted willingness to attack sensitive to the visible matchup.
    The returned index matches the requested battle order.
    """
    battle_indices = np.asarray(battle_indices)
    rows = 2 * battle_indices[:, None] + np.array([0, 1])[None, :]
    contexts = inputs.final.iloc[rows.ravel()].reset_index(drop=True)
    prob = model.predict_proba(contexts).reshape(-1, 2, 3)

    out = {}
    for side_idx, prefix in enumerate(("p1", "p2")):
        for k, name in enumerate(CLASSES):
            out[f"action_{prefix}_{name}_prob"] = prob[:, side_idx, k]

        offensive = contexts.iloc[side_idx::2]["own_best_revealed_power"].to_numpy()
        defensive = contexts.iloc[side_idx::2]["enemy_best_revealed_power"].to_numpy()
        out[f"action_{prefix}_expected_pressure"] = prob[:, side_idx, 2] * offensive
        out[f"action_{prefix}_expected_exposure"] = prob[:, 1-side_idx, 2] * defensive

    for j, name in enumerate(CLASSES):
        out[f"action_diff_{name}_prob"] = prob[:, 0, j] - prob[:, 1, j]

    out["action_diff_expected_pressure"] = (
        out["action_p1_expected_pressure"] - out["action_p2_expected_pressure"]
    )
    out["action_clash_probability"] = prob[:, 0, 2] * prob[:, 1, 2]
    out["action_both_passive_probability"] = (1-prob[:, 0, 2]) * (1-prob[:, 1, 2])

    return pd.DataFrame(out, dtype=np.float32)
