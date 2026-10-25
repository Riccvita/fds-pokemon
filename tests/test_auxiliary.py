"""Smoke tests for the self-supervised auxiliary feature extractors.

These tests use small fixtures built exclusively from the provided competition
format. They do not train on or inspect the final battle-outcome label.
"""

import unittest

import numpy as np

from pokemon.features import read_jsonl, load_type_chart
from pokemon.features2 import build_dex
from pokemon.forecast import build_forecast_inputs
from pokemon.transitions import build_transition_inputs


class AuxiliaryFeatureTests(unittest.TestCase):
    """Check shape, alignment, and independence from the winner label."""

    @classmethod
    def setUpClass(cls):
        """Load source data from an explicitly supplied test directory."""
        import os
        from pathlib import Path

        folder = os.getenv("POKEMON_DATA_DIR")
        if not folder:
            raise unittest.SkipTest("Set POKEMON_DATA_DIR to run data-dependent tests")

        folder = Path(folder)
        cls.battles = read_jsonl(folder / "train.jsonl")[:3]
        cls.dex = build_dex(cls.battles)
        cls.chart = load_type_chart(folder / "type_chart_gen2.csv")

    def test_transition_targets_do_not_use_winner(self):
        """Removing winner labels must not affect transition training rows."""
        without_labels = [{k: v for k, v in row.items() if k != "player_won"}
                          for row in self.battles]
        labeled = build_transition_inputs(self.battles, self.dex, self.chart)
        unlabeled = build_transition_inputs(without_labels, self.dex, self.chart)

        np.testing.assert_array_equal(labeled.target, unlabeled.target)
        self.assertEqual(labeled.matchup.shape, unlabeled.matchup.shape)
        self.assertEqual(len(labeled.fixed_features), len(self.battles))

    def test_forecast_targets_are_bounded_and_label_independent(self):
        """Each earlier snapshot predicts only a visible next-ten-turn delta."""
        without_labels = [{k: v for k, v in row.items() if k != "player_won"}
                          for row in self.battles]
        labeled = build_forecast_inputs(self.battles, self.dex)
        unlabeled = build_forecast_inputs(without_labels, self.dex)

        np.testing.assert_allclose(labeled.history_labels, unlabeled.history_labels)
        self.assertEqual(labeled.history.shape[0], 4 * len(self.battles))
        self.assertEqual(labeled.final.shape[0], len(self.battles))
        self.assertTrue(np.isfinite(labeled.history_labels.to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
