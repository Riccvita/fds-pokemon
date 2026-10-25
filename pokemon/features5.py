"""Temporal resource, per-species, and battle-pressure feature engineering.

The module uses only the supplied 30-turn battle timelines and the species
statistics and type chart provided in the competition files. All features are
available at prediction time and contain no outcome-dependent statistics.
"""

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from pokemon.features import STATS, type_mult


# A recorded major status remains relevant even after the Pokémon switches out.
PERSISTENT_STATUSES = ("slp", "par", "tox", "psn", "frz", "brn")
BOOSTS = ("atk", "def", "spa", "spd", "spe")


def _species_stats(name, dex):
    """Return the six intrinsic stats for a species found in the input data."""
    data = dex.get(name)
    return [float(data[stat]) for stat in STATS] if data else [np.nan] * len(STATS)


def _snapshot(states, full_team, dex):
    """Aggregate health and available resources from the latest known states.

    Unknown species count as 100% healthy; for player two their identities are
    unknown, so strength summaries only include observed members.
    """
    health = [float(state["hp_pct"]) for state in states.values()]
    status = [state["status"] for state in states.values()]
    hidden = max(0, 6 - len(states))

    if full_team is not None:
        all_names = full_team
    else:
        all_names = tuple(states)

    all_health = [float(states[name]["hp_pct"]) if name in states else 1.0
                  for name in all_names]

    total_health = sum(all_health) + (hidden if full_team is None else 0.0)
    health2 = sum(h * h for h in all_health) + (hidden if full_team is None else 0.0)
    alive = sum(h > 0.0 for h in all_health) + (hidden if full_team is None else 0)

    # Effective party HP is more informative than the active Pokémon's HP.
    results = {
        "hp_sum": total_health,
        "hp_squared": health2,
        "alive": alive,
        "seen": len(states),
        "unseen": hidden,
        "observed_hp_sum": sum(health),
        "observed_hp_min": min(health) if health else 1.0,
        "observed_hp_std": float(np.std(health)) if health else 0.0,
        "below_25": sum(0.0 < h < 0.25 for h in all_health),
        "below_50": sum(0.0 < h < 0.5 for h in all_health),
        "below_75": sum(0.0 < h < 0.75 for h in all_health),
        "knocked_out": sum(h == 0 for h in all_health),
        "paralyzed": status.count("par"),
        "asleep": status.count("slp"),
        "poisoned": status.count("psn") + status.count("tox"),
        "burned": status.count("brn"),
        "frozen": status.count("frz"),
    }

    # Damage-adjusted strengths of the known team members.
    for stat in STATS:
        strengths = [(dex[n][stat], float(states[n]["hp_pct"]))
                     for n in all_names if n in states and n in dex]
        if full_team is not None:
            strengths.extend((dex[n][stat], 1.0)
                             for n in full_team if n not in states and n in dex)

        results[f"alive_{stat}"] = sum(value for value, hp in strengths if hp > 0)
        results[f"hpweighted_{stat}"] = sum(value * hp for value, hp in strengths)

    results["hpweighted_bst"] = sum(results[f"hpweighted_{stat}"] for stat in STATS)

    return results


def _side_features(timeline, side, full_team, dex, chart):
    """Construct health trajectories, resource summaries and attacking signals."""
    state_field = f"{side}_pokemon_state"
    move_field = f"{side}_move_details"
    other = "p2" if side == "p1" else "p1"
    other_state_field = f"{other}_pokemon_state"

    states = {}
    first_hp = {}
    seen_at = {}
    hp_per_turn = []
    running_team_hp = 6.0
    recorded_states = []
    hp_loss = []
    hp_gain = []
    move_power = []
    move_eff = []
    move_hit_power = []
    move_types = Counter()
    move_categories = Counter()
    departures = []
    returns = 0
    new_species = 0
    entry_hp = []
    observed_damage = 0.0
    observed_healing = 0.0
    prev_name = None

    # --- Scan turns chronologically. Never use information after turn 30.
    for idx, turn in enumerate(timeline):
        state = turn[state_field]
        name = state["name"]
        hp = float(state["hp_pct"])
        move = turn[move_field]

        if name in states:
            old_hp = float(states[name]["hp_pct"])
            running_team_hp += hp - old_hp
            gain = max(hp - old_hp, 0.0)
            loss = max(old_hp - hp, 0.0)
            observed_healing += gain
            observed_damage += loss
            hp_gain.append(gain)
            hp_loss.append(loss)
        else:
            running_team_hp += hp - 1.0
            first_hp[name] = hp
            seen_at[name] = idx + 1
            new_species += 1
            hp_gain.append(0.0)
            hp_loss.append(0.0)

        if prev_name is not None and prev_name != name:
            departures.append((prev_name, idx))
            if name in states:
                returns += 1
            entry_hp.append(hp)

        states[name] = state
        prev_name = name
        recorded_states.append(state)

        # Approximate total party health at every observation, carrying forward
        # the last seen HP for inactive Pokémon instead of losing their state.
        hp_per_turn.append(running_team_hp)

        if move is not None:
            category = move["category"]
            move_categories[category] += 1
            move_types[move["type"]] += 1

            if category in ("physical", "special"):
                power = float(move["base_power"])
                move_power.append(power)
                opponent_name = turn[other_state_field]["name"]
                attacker = dex.get(name)
                defender = dex.get(opponent_name)

                if attacker and defender:
                    multiplier = type_mult(chart, move["type"], defender["types"])
                    stab = 1.5 if move["type"] in attacker["types"] else 1.0
                    effectiveness = multiplier * stab
                    move_eff.append(effectiveness)

                    attack_stat = attacker["atk" if category == "physical" else "spa"]
                    defense_stat = defender["def" if category == "physical" else "spd"]
                    move_hit_power.append(power * effectiveness * attack_stat / max(defense_stat, 1))

    end = _snapshot(states, full_team, dex)
    features = dict(end)
    features["last_species_base_speed"] = dex.get(prev_name, {}).get("spe", np.nan)

    # --- Health advantage trends: aggregate the WHOLE team, not active HP only.
    hp = np.asarray(hp_per_turn, dtype=float)
    for t in (5, 10, 15, 20, 25, 30):
        features[f"party_hp_t{t}"] = float(hp[t - 1])

    for k in (5, 10, 15):
        features[f"party_hp_delta{k}"] = float(hp[-1] - hp[-k - 1])
        features[f"party_hp_mean{k}"] = float(hp[-k:].mean())

    features["party_hp_slope"] = float(np.polyfit(np.arange(30), hp, 1)[0])
    features["party_hp_area"] = float(hp.mean())
    features["damage_observed"] = observed_damage
    features["healing_observed"] = observed_healing
    features["damage_last10"] = sum(hp_loss[-10:])
    features["heal_last10"] = sum(hp_gain[-10:])
    features["return_switches"] = returns
    features["unique_species"] = new_species
    features["entry_hp_mean"] = float(np.mean(entry_hp)) if entry_hp else 1.0
    features["initial_active_hp"] = float(recorded_states[0]["hp_pct"])

    # --- At battle end, the status of each previously seen Pokémon still matters.
    for status in PERSISTENT_STATUSES:
        features[f"statuses_{status}"] = sum(s["status"] == status for s in states.values())
        features[f"active_is_{status}"] = int(recorded_states[-1]["status"] == status)

    features["toxic_count"] = sum(s["status"] == "tox" for s in states.values())
    features["boost_positive"] = sum(max(v, 0) for v in recorded_states[-1]["boosts"].values())
    features["boost_negative"] = sum(min(v, 0) for v in recorded_states[-1]["boosts"].values())

    # --- Only moves actually visible in the log are considered.
    features["damaging_moves"] = len(move_power)
    features["damage_power_total"] = sum(move_power)
    features["damage_power_mean"] = float(np.mean(move_power)) if move_power else 0.0
    features["attack_effective_mean"] = float(np.mean(move_eff)) if move_eff else 1.0
    features["attack_effective_best"] = max(move_eff, default=1.0)
    features["attack_pressure_total"] = sum(move_hit_power)
    features["attack_pressure_last10"] = sum(move_hit_power[-10:])
    features["attack_pressure_max"] = max(move_hit_power, default=0.0)
    features["unique_move_types"] = len(move_types)
    features["unique_physical_special"] = move_categories["physical"] + move_categories["special"]

    return features, states, hp


def battle_features(battle, dex, chart, species_vocabulary):
    """Extract one battle's temporal and per-species predictive features.

    Args:
        battle: Competition JSON object with exactly the observed timeline.
        dex: Species stats dictionary derived from the provided battle records.
        chart: Official supplied Generation 2 type effectiveness table.
        species_vocabulary: Pokémon names receiving dedicated HP/status columns.

    Returns:
        Numerical, label-independent features as a dictionary.
    """
    timeline = battle["battle_timeline"]
    team1 = tuple(p["name"] for p in battle["p1_team_details"])
    feature_dict = {}

    # --- Independently summarize both sides, preserving an identical schema.
    side1, last1, h1 = _side_features(timeline, "p1", team1, dex, chart)
    side2, last2, h2 = _side_features(timeline, "p2", None, dex, chart)

    for name in side1:
        a, b = side1[name], side2[name]
        feature_dict[f"dyna_p1_{name}"] = a
        feature_dict[f"dyna_p2_{name}"] = b
        feature_dict[f"dyna_diff_{name}"] = a - b

    # --- Relative battle progress at selected turn checkpoints.
    health_delta = h1 - h2
    for t in (5, 10, 15, 20, 25, 30):
        feature_dict[f"dyna_adv_t{t}"] = float(health_delta[t - 1])

    for k in (5, 10, 15):
        feature_dict[f"dyna_adv_last{k}"] = float(health_delta[-k:].mean())
        feature_dict[f"dyna_adv_gain{k}"] = float(health_delta[-1] - health_delta[-k - 1])

    # --- Dedicated species-health columns and status flags.
    # Important: unseen P1 Pokémon are at full HP; unseen P2 species are unknown,
    # not zero HP. NaN marks the latter rather than implying that it fainted.
    for species in species_vocabulary:
        p1_member = species in team1
        s1 = last1.get(species)
        s2 = last2.get(species)

        feature_dict[f"dyna_p1_hp_{species}"] = (
            float(s1["hp_pct"]) if s1 is not None else 1.0 if p1_member else np.nan
        )
        feature_dict[f"dyna_p2_hp_{species}"] = float(s2["hp_pct"]) if s2 is not None else np.nan
        feature_dict[f"dyna_p1_status_{species}"] = (
            int(s1["status"] != "nostatus" and s1["status"] != "fnt") if s1 else 0
        ) if p1_member else np.nan
        feature_dict[f"dyna_p2_status_{species}"] = (
            int(s2["status"] != "nostatus" and s2["status"] != "fnt") if s2 else np.nan
        )

    return feature_dict


def build_temporal_features(battles, dex, chart, vocabulary=None, min_frequency=100):
    """Build feature matrix with a vocabulary fixed from training input only.

    Args:
        battles: Sequence of raw battle dictionaries.
        dex: Species base statistics from competition data only.
        chart: Supplied Generation 2 type chart.
        vocabulary: Optional fixed list of species for validation/test reuse.
        min_frequency: Minimum P1 team membership when building vocabulary.

    Returns:
        A tuple of (feature matrix, vocabulary).
    """
    if vocabulary is None:
        species_counts = Counter(
            pokemon["name"] for battle in battles for pokemon in battle["p1_team_details"]
        )
        vocabulary = tuple(sorted(n for n, c in species_counts.items() if c >= min_frequency))

    rows = [battle_features(battle, dex, chart, vocabulary) for battle in battles]

    return pd.DataFrame(rows, dtype=float), vocabulary
