"""Unit tests for added label-independent and cross-fitted feature builders."""

import unittest

import numpy as np

from pokemon.identity_features import build_identity_features, names_for_battle
from pokemon.pair_te import extract_rosters, oof_pair_features


class ResearchFeatureTests(unittest.TestCase):
    """Check schema consistency and prevent information leakage."""

    @staticmethod
    def _battle(index):
        """Construct a tiny battle without external Pokémon knowledge tables."""
        p1 = [{'name': f'left_{index}_{slot}'} for slot in range(6)]
        p2name = f'right_{index}'
        return {
            'player_won': index % 2,
            'p1_team_details': p1,
            'battle_timeline': [
                {
                    'p1_pokemon_state': {'name': p1[0]['name']},
                    'p2_pokemon_state': {'name': p2name},
                },
                {
                    'p1_pokemon_state': {'name': p1[1]['name']},
                    'p2_pokemon_state': {'name': p2name},
                },
            ],
        }

    def test_identity_features_use_observations_only(self):
        """Changing a target must not modify any categorical feature."""
        battles = [self._battle(i) for i in range(6)]
        first, vocabulary = build_identity_features(battles, minimum_count=1)

        for battle in battles:
            battle['player_won'] = 1 - battle['player_won']

        second, _ = build_identity_features(battles, vocabulary=vocabulary)
        self.assertTrue(first.equals(second))
        self.assertEqual(names_for_battle(battles[0])['p1_last'], 'left_0_1')

    def test_pair_targets_exclude_their_own_training_label(self):
        """Each training row is unique, so inner-OOF pair counts must be zero."""
        battles = [self._battle(i) for i in range(10)]
        rosters = extract_rosters(battles)
        labels = np.array([battle['player_won'] for battle in battles])

        train_features, validation_features = oof_pair_features(
            rosters, labels, np.arange(8), np.arange(8, 10),
        )

        self.assertEqual(len(train_features), 8)
        self.assertEqual(len(validation_features), 2)
        self.assertTrue(np.allclose(train_features['roster_p1_pair_weight'], 0))
        self.assertTrue(np.allclose(train_features['roster_p1_pair_mean'], 0.5))
        self.assertTrue(np.allclose(validation_features['roster_p1_pair_weight'], 0))


if __name__ == '__main__':
    unittest.main()
