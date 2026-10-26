"""Out-of-fold target encoding of team strength.

Each distinct team (set of six Pokemon) gets a smoothed win rate computed from the
labelled battles it appears in, as P1 (a win counts for the team) or as P2 (a P1
loss counts for the team). Statistics are always computed on data that excludes
the rows being encoded, so no label leaks into its own feature.
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedGroupKFold


def team_keys(battles):
    """Return (k1, k2): the team key of P1 and, when fully revealed, of P2.

    A key is the frozenset of the six Pokemon names. P2's key is None unless all six
    P2 Pokemon appear in the observed turns.
    """
    k1 = [frozenset(p["name"] for p in b["p1_team_details"]) for b in battles]

    k2 = []
    for b in battles:
        seen = frozenset(t["p2_pokemon_state"]["name"] for t in b["battle_timeline"])
        k2.append(seen if len(seen) == 6 else None)

    return k1, k2


def fit_stats(k1, k2, y, idx):
    """Accumulate win/game counts per team over the battles in `idx`.

    Each entry is [p1_wins, p1_games, p2_wins, p2_games]; P2 wins when P1 loses (y == 0).
    """
    stats = defaultdict(lambda: [0, 0, 0, 0])

    for i in idx:
        s = stats[k1[i]]
        s[1] += 1
        s[0] += y[i]

        if k2[i] is not None:
            s = stats[k2[i]]
            s[3] += 1
            s[2] += 1 - y[i]

    return stats


def apply_stats(stats, k1, k2, idx, K=3.0):
    """Encode the battles in `idx` with exact-match team statistics.

    Win rates are smoothed towards 0.5 with a prior worth K games. Unknown or
    unrevealed teams get the neutral value 0.5 and zero games.
    """
    rows = []

    for i in idx:
        row = []

        for key in (k1[i], k2[i]):
            if key is None or key not in stats:
                row += [0.5, 0, 0.5, 0]
            else:
                w1, n1, w2, n2 = stats[key]
                w, n = w1 + w2, n1 + n2
                row += [(w + 0.5 * K) / (n + K), n,        # overall win rate and games
                        (w1 + 0.5 * K) / (n1 + K), n1]     # win rate and games on this side only

        rows.append(row)

    cols = ["te_p1_wr", "te_p1_n", "te_p1_wr_asp1", "te_p1_n_asp1",
            "te_p2_wr", "te_p2_n", "te_p2_wr_asp2", "te_p2_n_asp2"]

    df = pd.DataFrame(np.array(rows), columns=cols)
    df["te_diff"] = df["te_p1_wr"] - df["te_p2_wr"]

    return df


def cand_te(stats, keys, cands, idx, K=3.0):
    """Average win rate of the pool teams compatible with P2's seen Pokemon.

    Unlike the exact-match encoding it also works when P2 has revealed fewer than six
    Pokemon. Returns the mean smoothed win rate and the mean number of games.
    """
    rows = []

    for i in idx:
        cand = cands[i]

        if len(cand) == 0:
            rows.append([0.5, 0.0])
            continue

        win_rates, games = [], []

        for j in cand:
            s = stats.get(keys[j])

            if s is None:
                win_rates.append(0.5)
                games.append(0)
            else:
                w, n = s[0] + s[2], s[1] + s[3]
                win_rates.append((w + 0.5 * K) / (n + K))
                games.append(n)

        rows.append([float(np.mean(win_rates)), float(np.mean(games))])

    return pd.DataFrame(rows, columns=["te_p2c_wr", "te_p2c_n"])


def _encode(stats, k1, k2, keys, cands, idx):
    """Exact-match and candidate-average encodings for `idx`, plus their difference."""
    df = pd.concat([apply_stats(stats, k1, k2, idx), cand_te(stats, keys, cands, idx)], axis=1)
    df["te_diff2"] = df["te_p1_wr"] - df["te_p2c_wr"]

    return df


def oof_te2(k1, k2, y, keys, cands, train_idx, apply_idx, n_splits=5, seed=0, groups=None):
    """Target-encode a training set out-of-fold and a second set with the full training stats.

    Rows in `train_idx` are encoded with statistics from the other inner folds only.
    Rows in `apply_idx` (validation or test) are encoded with the statistics of the
    whole `train_idx`. `apply_idx` may be None, in which case only the first frame is built.

    Optional groups keep duplicate battles or P1 composition components together
    in the inner encoder splits. Fully revealed P2 occurrences are still counted
    from fitting rows; P1 grouping does not imply absence from both sides.
    Returns (train_features, apply_features), both with a fresh 0..n-1 index.
    """
    train_idx = np.asarray(train_idx)
    parts = []

    # --- out-of-fold encoding of the training rows
    if groups is None:
        splits = KFold(n_splits, shuffle=True, random_state=seed).split(train_idx)
    else:
        local_groups = np.asarray(groups)[train_idx]
        folds = min(n_splits, len(set(local_groups)))
        if folds < 2:
            raise ValueError("Target encoding needs at least two distinct training groups.")
        splits = StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(
            train_idx, np.asarray(y)[train_idx], local_groups,
        )
    for fit_pos, enc_pos in splits:
        if groups is not None and set(local_groups[fit_pos]) & set(local_groups[enc_pos]):
            raise AssertionError("Target encoding split crosses protected groups")
        stats = fit_stats(k1, k2, y, train_idx[fit_pos])

        part = _encode(stats, k1, k2, keys, cands, train_idx[enc_pos])
        part.index = enc_pos
        parts.append(part)

    train_features = pd.concat(parts).sort_index().reset_index(drop=True)

    # --- encoding of the held-out / test rows with the full training statistics
    apply_features = None

    if apply_idx is not None:
        stats = fit_stats(k1, k2, y, train_idx)
        apply_features = _encode(stats, k1, k2, keys, cands, apply_idx)

    return train_features, apply_features
