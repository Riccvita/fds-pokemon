"""Categorical context for the opening and final observed battle positions.

Species identities are derived only from the first 30 recorded turns. The
vocabulary is fit on the provided training data without using target labels.
"""

from collections import Counter

import pandas as pd


POSITIONS = ('p1_first', 'p2_first', 'p1_last', 'p2_last')


def names_for_battle(battle):
    """Extract lead and final-active species names for the two players."""
    turns = battle['battle_timeline']

    return {
        'p1_first': turns[0]['p1_pokemon_state']['name'],
        'p2_first': turns[0]['p2_pokemon_state']['name'],
        'p1_last': turns[-1]['p1_pokemon_state']['name'],
        'p2_last': turns[-1]['p2_pokemon_state']['name'],
    }


def build_identity_features(battles, vocabulary=None, minimum_count=25):
    """One-hot encode the opening and final active positions.

    Parameters
    ----------
    battles : list[dict]
        Provided competition battles, truncated at 30 observed turns.
    vocabulary : dict[str, tuple[str, ...]] or None
        Training-derived vocabulary, reused for held-out data.
    minimum_count : int
        Minimum observed species count to create a category column.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        Numeric feature matrix and its training-only category vocabulary.
    """
    records = [names_for_battle(battle) for battle in battles]

    # --- Never use player_won to select or build feature columns.
    if vocabulary is None:
        vocabulary = {}

        for position in POSITIONS:
            counts = Counter(record[position] for record in records)
            vocabulary[position] = tuple(sorted(
                name for name, count in counts.items() if count >= minimum_count
            ))

    encoded = {}

    # --- Keep the same column order for the training and test partitions.
    for position in POSITIONS:
        for species in vocabulary[position]:
            encoded[f'identity_{position}_{species}'] = [
                int(record[position] == species) for record in records
            ]

    return pd.DataFrame(encoded, dtype='int8'), vocabulary
