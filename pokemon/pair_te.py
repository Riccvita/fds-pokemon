"""Leakage-safe species, teammate-pair, and cross-team matchup target encodings.

All statistics come from labeled competition training rows. Training features
are computed out of fold; held-out features use only outer-training labels.
Only Pokémon actually observed during the first 30 turns are used for P2.
"""

from collections import defaultdict
from itertools import combinations, product

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


COLUMNS = [
    'roster_p1_mon_mean', 'roster_p2_mon_mean', 'roster_mon_diff',
    'roster_p1_mon_weight', 'roster_p2_mon_weight',
    'roster_p1_pair_mean', 'roster_p2_pair_mean', 'roster_pair_diff',
    'roster_p1_pair_weight', 'roster_p2_pair_weight',
    'roster_p1_pair_max', 'roster_p2_pair_max',
    'roster_p1_pair_min', 'roster_p2_pair_min',
    'versus_mean', 'versus_min', 'versus_max', 'versus_n',
    'versus_weight', 'roster_p2_observed_n',
]


def extract_rosters(battles):
    """Return observed team identities, never using labels or hidden turns."""
    result = []

    for battle in battles:
        p1 = tuple(sorted({member['name'] for member in battle['p1_team_details']}))
        p2 = tuple(sorted({turn['p2_pokemon_state']['name']
                           for turn in battle['battle_timeline']}))
        result.append((p1, p2))

    return result


def _pairs(pokemon):
    """Return all unordered pairs from one observed roster."""
    return list(combinations(pokemon, 2))


def _fit(rosters, labels, indices):
    """Aggregate supervised team and matchup counts exclusively on fit indices."""
    single = defaultdict(lambda: [0.0, 0.0])
    pair = defaultdict(lambda: [0.0, 0.0])
    versus = defaultdict(lambda: [0.0, 0.0])

    for index in indices:
        left, right = rosters[index]
        winner = float(labels[index])

        for roster, won in ((left, winner), (right, 1.0 - winner)):
            for mon in roster:
                single[mon][0] += won
                single[mon][1] += 1

            for combo in _pairs(roster):
                pair[combo][0] += won
                pair[combo][1] += 1

        for left_mon, right_mon in product(left, right):
            versus[(left_mon, right_mon)][0] += winner
            versus[(left_mon, right_mon)][1] += 1
            versus[(right_mon, left_mon)][0] += 1.0 - winner
            versus[(right_mon, left_mon)][1] += 1

    return single, pair, versus


def _summarize(keys, stats, shrinkage):
    """Summarize regularized win rates and evidence for a collection of keys."""
    if not keys:
        return 0.5, 0.0, 0.5, 0.5, 0

    rates = []
    weights = []

    for key in keys:
        wins, games = stats.get(key, (0.0, 0.0))
        rates.append((wins + 0.5 * shrinkage) / (games + shrinkage))
        weights.append(games)

    return (float(np.mean(rates)), float(np.mean(weights)),
            float(np.min(rates)), float(np.max(rates)), len(keys))


def _apply(rosters, indices, fitted, prior_single=30.0,
           prior_pair=20.0, prior_versus=40.0):
    """Encode held-out battles using statistics from separate labeled rows."""
    single, pair, versus = fitted
    encoded = []

    for index in indices:
        left, right = rosters[index]
        a = _summarize(left, single, prior_single)
        b = _summarize(right, single, prior_single)
        c = _summarize(_pairs(left), pair, prior_pair)
        d = _summarize(_pairs(right), pair, prior_pair)
        e = _summarize(list(product(left, right)), versus, prior_versus)

        encoded.append([
            a[0], b[0], a[0] - b[0], a[1], b[1],
            c[0], d[0], c[0] - d[0], c[1], d[1],
            c[3], d[3], c[2], d[2],
            e[0], e[2], e[3], e[4], e[1], len(right),
        ])

    return pd.DataFrame(encoded, columns=COLUMNS)


def oof_pair_features(rosters, labels, train_idx, apply_idx, seed=42):
    """Cross-fit training encodings and independently encode held-out examples.

    Parameters
    ----------
    rosters : list[tuple[tuple[str, ...], tuple[str, ...]]]
        Competition-only species identities for both battle sides.
    labels : array-like
        Training labels, never the Kaggle test set.
    train_idx : array-like
        Outer-fold training row positions.
    apply_idx : array-like
        Held-out validation or unlabeled Kaggle test row positions.
    seed : int
        Fixed reproducibility seed for inner cross-fitting.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Inner-OOF training features and independently derived held-out features.
    """
    train_idx = np.asarray(train_idx)
    apply_idx = np.asarray(apply_idx)
    encoded_parts = []
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    # --- Strictly cross-fit supervised statistics for training features.
    for fit_pos, encode_pos in cv.split(train_idx):
        fitted = _fit(rosters, labels, train_idx[fit_pos])
        values = _apply(rosters, train_idx[encode_pos], fitted)
        values.index = encode_pos
        encoded_parts.append(values)

    training = pd.concat(encoded_parts).sort_index().reset_index(drop=True)

    # --- Validation and test feature construction uses all training labels only.
    fitted = _fit(rosters, labels, train_idx)
    held_out = _apply(rosters, apply_idx, fitted)

    return training, held_out
