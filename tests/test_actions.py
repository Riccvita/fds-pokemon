"""Behavioral tests for the label-free, causal action feature pipeline."""

import copy
import unittest

import numpy as np

from pokemon.actions import (
    CUTOFFS, _action_code, action_features,
    build_action_inputs, fit_action_model,
)


class ActionFeatureTests(unittest.TestCase):
    """Ensure labels and features respect the observed prefix boundary."""

    @staticmethod
    def example_battle(identifier, attack_every=2):
        """Construct a small deterministic synthetic 30-turn battle."""
        moves = (
            {"name": "testmove", "type": "normal", "category": "physical",
             "base_power": 60, "accuracy": 1.0, "priority": 0},
            {"name": "teststatus", "type": "normal", "category": "status",
             "base_power": 0, "accuracy": 1.0, "priority": 0},
        )
        timeline = []

        for idx in range(30):
            st1 = {"name": "alpha", "hp_pct": 1.0 - .01 * idx,
                   "status": "nostatus", "effects": ["noeffect"],
                   "boosts": {stat: 0 for stat in ("atk", "def", "spa", "spd", "spe")}}
            st2 = {"name": "beta", "hp_pct": 1.0 - .02 * idx,
                   "status": "nostatus", "effects": ["noeffect"],
                   "boosts": {stat: 0 for stat in ("atk", "def", "spa", "spd", "spe")}}

            timeline.append({
                "turn": idx + 1, "p1_pokemon_state": st1,
                "p2_pokemon_state": st2,
                "p1_move_details": moves[idx % 2] if idx % 3 != 0 else None,
                "p2_move_details": moves[(idx + 1) % 2] if idx % attack_every else None,
            })

        return {"battle_id": identifier, "battle_timeline": timeline,
                "p1_team_details": [{"name": "alpha"}]}

    def setUp(self):
        """Initialize the competition-derived schema with miniature species."""
        self.dex = {
            species: {"hp": 100, "atk": 100, "def": 100, "spa": 100,
                      "spd": 100, "spe": 100, "types": ["normal", "notype"]}
            for species in ("alpha", "beta")
        }
        self.chart = {("normal", "normal"): 1.0}

    def test_action_classes(self):
        """Unrecorded actions are not falsely labeled as switches."""
        self.assertEqual(_action_code(None), 0)
        self.assertEqual(_action_code({"category": "status"}), 1)
        self.assertEqual(_action_code({"category": "physical"}), 2)
        self.assertEqual(_action_code({"category": "special"}), 2)

    def test_future_action_does_not_change_past_features(self):
        """Changing turn 30 must not affect features from cutoff 29."""
        battle = self.example_battle("a")
        modified = copy.deepcopy(battle)
        modified["battle_timeline"][-1]["p1_move_details"] = None

        original = build_action_inputs([battle], self.dex, self.chart)
        changed = build_action_inputs([modified], self.dex, self.chart)

        np.testing.assert_array_equal(original.history, changed.history)
        self.assertNotEqual(original.history_target[-2], changed.history_target[-2])
        self.assertEqual(original.final.shape, (2, original.history.shape[1]))
        self.assertEqual(len(original.history), 2 * len(CUTOFFS))

    def test_group_indices_and_action_probability_shapes(self):
        """Prediction output includes only probabilities for requested battles."""
        battles = [self.example_battle(str(i), 2 + i % 4) for i in range(18)]
        inputs = build_action_inputs(battles, self.dex, self.chart)
        model = fit_action_model(inputs, np.arange(15))
        features = action_features(model, inputs, [16, 17])

        self.assertEqual(features.shape[0], 2)
        self.assertTrue(np.isfinite(features.to_numpy()).all())
        for prefix in ("p1", "p2"):
            columns = [f"action_{prefix}_{label}_prob"
                       for label in ("unrecorded", "status", "damage")]
            np.testing.assert_allclose(features[columns].sum(axis=1), 1.0)


if __name__ == "__main__":
    unittest.main()
