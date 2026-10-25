"""Behavioral tests for deterministic active-strength interactions."""

import unittest

import numpy as np
import pandas as pd

from pokemon.strength_features import build_strength_features


class StrengthFeatureTests(unittest.TestCase):
    """Check stat stages, status effects, missing data and row alignment."""

    @staticmethod
    def frame():
        """Make equal active Pokemon with non-consecutive row indices."""
        values = {}
        for side in ("p1", "p2"):
            values[f"{side}_active_hp"] = [1.0, 0.0]
            values[f"{side}_active_maxhp"] = [400.0, 400.0]
            values[f"{side}_active_brn"] = [0, 0]
            values[f"{side}_active_par"] = [0, 0]
            for stat in ("atk", "def", "spa", "spd", "spe"):
                values[f"{side}_act_{stat}"] = [100.0, 100.0]
                values[f"{side}_boost_{stat}"] = [0, 0]
        return pd.DataFrame(values, index=[19, 7])

    def test_target_and_row_order_independence(self):
        frame = self.frame()
        frame["player_won"] = [0, 1]
        original = frame.copy(deep=True)
        features = build_strength_features(frame)
        pd.testing.assert_frame_equal(frame, original)
        frame["player_won"] = [1, 0]
        pd.testing.assert_frame_equal(features, build_strength_features(frame))
        pd.testing.assert_frame_equal(
            features.iloc[::-1], build_strength_features(frame.iloc[::-1]),
        )
        self.assertEqual(features.shape, (2, 42))
        self.assertTrue(np.isfinite(features.to_numpy()).all())

    def test_stage_and_status_interactions(self):
        frame = self.frame()
        frame.loc[19, "p1_boost_atk"] = 2
        frame.loc[19, "p1_active_brn"] = 1
        frame.loc[19, "p1_boost_def"] = -2
        frame.loc[19, "p1_active_par"] = 1
        frame.loc[19, "p2_boost_spa"] = 6
        features = build_strength_features(frame)
        self.assertEqual(features.loc[19, "p1_effective_atk"], 300.0)
        self.assertEqual(features.loc[19, "p1_effective_def"], 150.0)
        self.assertEqual(features.loc[19, "p1_effective_spe"], 75.0)
        self.assertEqual(features.loc[19, "p2_effective_spa"], 1200.0)
        self.assertEqual(features.loc[7, "p1_physical_bulk"], 0.0)

    def test_unknown_species_remains_unknown(self):
        frame = self.frame()
        frame.loc[19, "p1_act_atk"] = np.nan
        features = build_strength_features(frame)
        self.assertTrue(np.isnan(features.loc[19, "p1_effective_atk"]))
        self.assertTrue(np.isnan(features.loc[19, "delta_effective_atk"]))


if __name__ == "__main__":
    unittest.main()
