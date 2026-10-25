"""Learn observable turn transitions and estimate endgame damage pressure.

Every training target in this module is a change in visible Pokémon health
between two observed turns. Final battle outcomes are never used to fit the
transition learner. All intrinsic species stats and moves come from the supplied
competition records; the supplied Generation 2 type chart is the only lookup.

The learner is intentionally a classical gradient-boosting regressor, not a
neural network or a pretrained simulator. Its outputs are estimates rather
than guaranteed game-mechanic damage calculations.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from pokemon.features import type_mult


COLUMNS = (
    "move_power", "move_accuracy", "move_priority", "move_physical",
    "move_special", "type_multiplier", "stab", "move_strength",
    "attacker_hp", "defender_hp", "attacker_speed", "defender_speed",
    "attack_stat", "defense_stat", "stat_ratio", "atk_boost",
    "def_boost", "attacker_status", "defender_status", "attacker_paralyzed",
    "defender_paralyzed", "attacker_burned", "defender_burned",
    "defender_poisoned", "defender_asleep", "turn_fraction", "attacker_is_p1",
)


@dataclass
class TransitionInputs:
    """Data for the damage learner and hypothetical matchup aggregator.

    Attributes
    ----------
    observed : pandas.DataFrame
        Attack situations with measurable next-state health changes.
    target : numpy.ndarray
        Observed net defender health decrease at consecutive turns.
    observed_battle : numpy.ndarray
        Source battle index of each self-supervised training example.
    matchup : pandas.DataFrame
        Hypothetical remaining-roster attacks with observed moves.
    matchup_battle : numpy.ndarray
        Source battle index for each hypothetical matchup.
    matchup_side : numpy.ndarray
        Attacking side, encoded as 0 for P1 and 1 for P2.
    matchup_group : numpy.ndarray
        Attacker species index within each battle and side.
    matchup_defender : numpy.ndarray
        Defender species index within each battle and side.
    matchup_defender_hp : numpy.ndarray
        Last observed health of the target Pokémon (fraction).
    matchup_is_active : numpy.ndarray
        Whether both participants are active in the final observed turn.
    fixed_features : pandas.DataFrame
        Features that do not require fitting the transition model.
    """

    observed: pd.DataFrame
    target: np.ndarray
    observed_battle: np.ndarray
    matchup: pd.DataFrame
    matchup_battle: np.ndarray
    matchup_side: np.ndarray
    matchup_group: np.ndarray
    matchup_defender: np.ndarray
    matchup_defender_hp: np.ndarray
    matchup_is_active: np.ndarray
    fixed_features: pd.DataFrame


def _stat_boost(value):
    """Convert an observed stat stage into a simplified multiplicative factor."""
    return (2 + value) / 2 if value >= 0 else 2 / (2 - value)


def _predictor_row(attacker, defender, move, dex, chart, turn, is_p1):
    """Encode one move/defender combination without using its outcome.

    Missing species are skipped instead of filling their stats with external
    information. The features deliberately approximate rather than reproduce
    every Gen 2 damage calculation.
    """
    attacker_data = dex.get(attacker["name"])
    defender_data = dex.get(defender["name"])

    if not attacker_data or not defender_data or not move or move["base_power"] <= 0:
        return None

    physical = move["category"] == "physical"
    attack_key, defense_key = ("atk", "def") if physical else ("spa", "spd")
    attack = attacker_data[attack_key]
    defense = defender_data[defense_key]
    atk_boost = attacker["boosts"][attack_key]
    def_boost = defender["boosts"][defense_key]
    mult = type_mult(chart, move["type"], defender_data["types"])
    stab = 1.5 if move["type"] in attacker_data["types"] else 1.0

    # The pressure proxy is based entirely on observable move and base stats.
    strength = move["base_power"] * mult * stab * attack / max(defense, 1)
    ratio = attack * _stat_boost(atk_boost) / max(defense * _stat_boost(def_boost), 1)

    astatus, dstatus = attacker["status"], defender["status"]
    return (
        float(move["base_power"]), float(move["accuracy"]),
        float(move["priority"]), float(physical),
        float(move["category"] == "special"), mult, stab, strength,
        float(attacker["hp_pct"]), float(defender["hp_pct"]),
        float(attacker_data["spe"]), float(defender_data["spe"]),
        float(attack), float(defense), ratio,
        float(atk_boost), float(def_boost),
        float(astatus != "nostatus" and astatus != "fnt"),
        float(dstatus != "nostatus" and dstatus != "fnt"),
        float(astatus == "par"), float(dstatus == "par"),
        float(astatus == "brn"), float(dstatus == "brn"),
        float(dstatus in ("psn", "tox")),
        float(dstatus == "slp"), float(turn) / 30.0, float(is_p1),
    )


def build_transition_inputs(battles, dex, chart):
    """Create label-free transition and counterfactual matchup tables.

    Parameters
    ----------
    battles : list[dict]
        Training or test battles with 30 observed turns.
    dex : dict
        Species statistics inferred from the supplied competition files.
    chart : dict
        Supplied Generation 2 type chart.

    Returns
    -------
    TransitionInputs
        Tables retaining battle indices for group-safe training and evaluation.

    Notes
    -----
    Only non-switching defenders yield reliable adjacent-turn health targets.
    The move from the CURRENT turn corresponds to the transition from the
    previous turn to the current turn in these data.
    """
    observed_rows, observed_targets, observed_battles = [], [], []
    matchup_rows, matchup_battles, matchup_sides = [], [], []
    matchup_groups, matchup_defenders, matchup_health, matchup_active = [], [], [], []
    fixed_rows = []

    for battle_index, battle in enumerate(battles):
        timeline = battle["battle_timeline"]
        team_p1 = [p["name"] for p in battle["p1_team_details"]]
        last_state = [dict(), dict()]
        moves_seen = [defaultdict(dict), defaultdict(dict)]
        active_names = [timeline[-1][f"p{j}_pokemon_state"]["name"] for j in (1, 2)]

        # --- Extract the supervised *transition* target, not the match winner.
        for turn_index, turn in enumerate(timeline):
            for side_index in (0, 1):
                side = f"p{side_index + 1}"
                state = turn[f"{side}_pokemon_state"]
                last_state[side_index][state["name"]] = state
                move = turn[f"{side}_move_details"]

                if move and move["base_power"] > 0:
                    moves_seen[side_index][state["name"]][move["name"]] = move

                if turn_index == 0:
                    continue

                previous = timeline[turn_index - 1]
                opponent_index = 1 - side_index
                opponent = f"p{opponent_index + 1}"
                defender = turn[f"{opponent}_pokemon_state"]
                old_defender = previous[f"{opponent}_pokemon_state"]

                # Only same-species consecutive states have a measurable delta.
                if defender["name"] != old_defender["name"]:
                    continue

                candidate = _predictor_row(
                    state, old_defender, move, dex, chart,
                    turn_index + 1, side_index == 0,
                )

                if candidate is not None:
                    observed_rows.append(candidate)
                    observed_targets.append(float(old_defender["hp_pct"] - defender["hp_pct"]))
                    observed_battles.append(battle_index)

        # --- Build remaining rosters at the observation cut-off.
        p2_seen = list(last_state[1])
        roster = [team_p1, p2_seen]
        alive = [[name for name in team if last_state[side].get(name, {}).get("hp_pct", 1) > 0]
                 for side, team in enumerate(roster)]
        n_missing_p2 = max(0, 6 - len(p2_seen))
        fixed = {
            "transition_p1_attacking_species": sum(bool(moves_seen[0][p]) for p in alive[0]),
            "transition_p2_attacking_species": sum(bool(moves_seen[1][p]) for p in alive[1]),
            "transition_p2_hidden_count": n_missing_p2,
        }

        # --- Evaluate observed attacking moves against *remaining* enemies.
        # Unknown P2 members stay unknown rather than being incorrectly set dead.
        for side_index in (0, 1):
            opposing = 1 - side_index
            available_moves = moves_seen[side_index]
            attackers = alive[side_index]
            defenders = alive[opposing]
            for member_index, attacker_name in enumerate(attackers):
                attacker = last_state[side_index].get(attacker_name)
                if attacker is None:
                    # An unseen P1 Pokémon's moves are unknown in this match.
                    continue

                attacks = list(available_moves[attacker_name].values())
                for defender_index, defender_name in enumerate(defenders):
                    defender = last_state[opposing].get(defender_name)
                    if defender is None:
                        # For P1 this means an unseen healthy member.
                        defender = {
                            "name": defender_name,
                            "hp_pct": 1.0,
                            "status": "nostatus",
                            "boosts": {stat: 0 for stat in ("atk", "def", "spa", "spd", "spe")},
                        }

                    for move in attacks:
                        encoded = _predictor_row(
                            attacker, defender, move, dex, chart, 30, side_index == 0,
                        )
                        if encoded is None:
                            continue

                        matchup_rows.append(encoded)
                        matchup_battles.append(battle_index)
                        matchup_sides.append(side_index)
                        matchup_groups.append(member_index)
                        matchup_defenders.append(defender_index)
                        matchup_health.append(float(defender["hp_pct"]))
                        matchup_active.append(
                            int(attacker_name == active_names[side_index]
                                and defender_name == active_names[opposing])
                        )

        fixed_rows.append(fixed)

    return TransitionInputs(
        observed=pd.DataFrame(observed_rows, columns=COLUMNS, dtype=np.float32),
        target=np.asarray(observed_targets, dtype=np.float32),
        observed_battle=np.asarray(observed_battles, dtype=np.int32),
        matchup=pd.DataFrame(matchup_rows, columns=COLUMNS, dtype=np.float32),
        matchup_battle=np.asarray(matchup_battles, dtype=np.int32),
        matchup_side=np.asarray(matchup_sides, dtype=np.int8),
        matchup_group=np.asarray(matchup_groups, dtype=np.int8),
        matchup_defender=np.asarray(matchup_defenders, dtype=np.int8),
        matchup_defender_hp=np.asarray(matchup_health, dtype=np.float32),
        matchup_is_active=np.asarray(matchup_active, dtype=np.int8),
        fixed_features=pd.DataFrame(fixed_rows).astype(np.float32),
    )


def train_transition_model(inputs, train_battle_indices, seed=42):
    """Fit only on observable transitions belonging to the outer training fold."""
    mask = np.isin(inputs.observed_battle, train_battle_indices)
    model = LGBMRegressor(
        n_estimators=130,
        learning_rate=0.045,
        num_leaves=12,
        max_depth=5,
        min_child_samples=90,
        reg_lambda=10.0,
        colsample_bytree=0.85,
        subsample=0.85,
        subsample_freq=1,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )
    model.fit(inputs.observed.loc[mask], inputs.target[mask])
    return model


def transition_features(model, inputs, indices, batch_size=150_000):
    """Summarize predicted damage pressure and likely knockout opportunities.

    The output has a row per requested battle and can be concatenated with the
    original feature matrix. The average includes only observed attacker moves
    and revealed opponent Pokémon, with corresponding coverage indicators.
    """
    wanted = np.asarray(indices, dtype=np.int32)
    chosen = np.isin(inputs.matchup_battle, wanted)
    loc = np.flatnonzero(chosen)

    predicted = np.empty(len(loc), dtype=np.float32)
    for start in range(0, len(loc), batch_size):
        part = loc[start:start + batch_size]
        predicted[start:start + len(part)] = model.predict(inputs.matchup.iloc[part])

    # Respect the 0..1 health range when using net HP changes as estimates.
    predicted = np.clip(predicted, 0.0, 1.0)
    table = pd.DataFrame({
        "battle": inputs.matchup_battle[loc],
        "side": inputs.matchup_side[loc],
        "member": inputs.matchup_group[loc],
        "defender": inputs.matchup_defender[loc],
        "damage": predicted,
        "defender_hp": inputs.matchup_defender_hp[loc],
        "active": inputs.matchup_is_active[loc],
    })
    table["can_knockout"] = (table["damage"] >= table["defender_hp"]).astype(float)
    table["damage_over_hp"] = np.minimum(table["damage"] / np.maximum(table["defender_hp"], 0.05), 5.0)

    # --- Compress move choices into the best observed move per matchup.
    # With unavailable moves or unrevealed opponent species, no prediction is made.
    per_target = table.groupby(["battle", "side", "member", "defender"], sort=False).agg(
        best_damage=("damage", "max"),
        best_hp_ratio=("damage_over_hp", "max"),
        knockout=("can_knockout", "max"),
    ).reset_index()

    summaries = {}
    for side in (0, 1):
        subset = table[table["side"] == side]
        targets = per_target[per_target["side"] == side]
        primary = subset[subset["active"] == 1]
        group = targets.groupby("battle", sort=False)
        p = f"transition_p{side + 1}_"
        summaries[p + "damage_mean"] = group["best_damage"].mean()
        summaries[p + "damage_max"] = group["best_damage"].max()
        summaries[p + "damage_sum"] = group["best_damage"].sum()
        summaries[p + "knockout_fraction"] = group["knockout"].mean()
        summaries[p + "hp_pressure"] = group["best_hp_ratio"].mean()
        summaries[p + "matchups_known"] = group.size().astype(float)
        summaries[p + "active_pressure"] = primary.groupby("battle")["damage"].max()
        summaries[p + "attacker_coverage"] = group["member"].nunique().astype(float)

    frame = pd.DataFrame(index=pd.Index(wanted, name="battle"))
    for name, grouped in summaries.items():
        frame[name] = grouped
    frame = frame.reset_index(drop=True)

    for name in ("damage_mean", "damage_max", "damage_sum", "knockout_fraction",
                 "hp_pressure", "active_pressure", "attacker_coverage"):
        a, b = f"transition_p1_{name}", f"transition_p2_{name}"
        frame[f"transition_diff_{name}"] = frame[a] - frame[b]

    result = pd.concat([
        inputs.fixed_features.iloc[wanted].reset_index(drop=True), frame,
    ], axis=1)

    return result.astype(np.float32)
