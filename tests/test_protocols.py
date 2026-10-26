"""Synthetic boundary tests; no real competition data or scores required."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from pokemon.features import STATS, BOOSTS
from pokemon.main import (VARIANTS, _validation_matrices, build_record_features,
                          compare_preparation, cross_validate, repeat_validation)
from pokemon.protocols import audit_battles, protocol_groups, split_indices
from pokemon.te import oof_te2, team_keys
from pokemon.validation import compare_oof


def battle(index, team=None, opponent=None):
    team = index // 2 if team is None else team
    opponent = (team + 1) % 15 if opponent is None else opponent

    def mon(name):
        return {"name": name, "types": ["normal", "notype"], "item": "leftovers",
                **{f"base_{stat}": 70 + (index % 3) for stat in STATS}}

    def state(name):
        return {"name": name, "hp_pct": 1 - index / 1000, "status": "nostatus",
                "boosts": {stat: 0 for stat in BOOSTS}, "effects": []}

    move = {"name": "tackle", "category": "physical", "base_power": 40}
    return {
        "battle_id": f"{index:04}", "player_won": index % 2,
        "p1_team_details": [mon(f"team{team}_{i}") for i in range(6)],
        "p2_lead_details": mon(f"team{opponent}_0"),
        "battle_timeline": [{"turn": i + 1,
                             "p1_pokemon_state": state(f"team{team}_{i % 6}"),
                             "p2_pokemon_state": state(f"team{opponent}_{i % 6}"),
                             "p1_move_details": move.copy(), "p2_move_details": move.copy()}
                            for i in range(30)],
    }


def dataset():
    train = [battle(i) for i in range(30)]
    test = [battle(31, team=100, opponent=101)]
    chart = {("normal", "normal"): 1.0}
    data = build_record_features(train, test, chart)
    audit, meta = audit_battles(train, test)
    data.update(train_records=train, test_records=test, chart=chart,
                audit=audit, audit_metadata=meta)
    return data


class TinyEstimator:
    """Deterministic fit-dependent predictions for orchestration tests."""
    def __init__(self, name, seed=42):
        self.name, self.seed = name, seed

    def fit(self, X, y):
        self.probability = float(np.mean(y)) + {"lgb": -0.05, "xgb": 0.05, "cat": 0}[self.name]

    def predict_proba(self, X):
        p = np.full(len(X), self.probability)
        return np.column_stack([1 - p, p])

    def get_params(self):
        return {"seed": self.seed}


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = dataset()

    def test_original_splits_unchanged_and_seeds_separate(self):
        data = self.data
        expected = list(StratifiedKFold(5, shuffle=True, random_state=42).split(data["A"], data["y"]))
        actual = split_indices(data["y"], np.arange(30))
        for (a, b), (c, d) in zip(expected, actual):
            np.testing.assert_array_equal(a, c)
            np.testing.assert_array_equal(b, d)
        with tempfile.TemporaryDirectory() as folder, patch("pokemon.main.make_model", side_effect=TinyEstimator):
            summary = repeat_validation(data, folder)
            self.assertEqual(summary["seeds"], [42, 123, 2026])
            folds = []
            for seed in summary["seeds"]:
                path = Path(folder) / f"seed_{seed}"
                report = json.loads((path / "validation_report.json").read_text())
                self.assertEqual((report["model_seed"], report["te_seed"]), (42, 0))
                folds.append(pd.read_csv(path / "validation_predictions.csv")["fold"].tolist())
            self.assertNotEqual(folds[0], folds[1])
            self.assertNotIn("best_seed", summary)

    def test_duplicate_components_preserved_without_deleting_rows(self):
        train = copy.deepcopy(self.data["train_records"])
        duplicate = copy.deepcopy(train[0])
        duplicate["battle_id"], duplicate["player_won"] = "new_id", 1
        train.append(duplicate)
        # Same ID with a different observation links transitively.
        train[1]["battle_id"] = "new_id"
        audit, meta = audit_battles(train)
        self.assertEqual(audit["rows_removed"], 0)
        self.assertEqual(meta["duplicate_groups"][0], meta["duplicate_groups"][1])
        self.assertEqual(meta["duplicate_groups"][0], meta["duplicate_groups"][-1])
        self.assertTrue(audit["conflicting_label_components"])
        y = np.array([b["player_won"] for b in train])
        for fit, valid in split_indices(y, np.arange(len(y)), groups=meta["duplicate_groups"]):
            self.assertFalse(set(meta["duplicate_groups"][fit]) & set(meta["duplicate_groups"][valid]))
        self.assertNotEqual(meta["duplicate_groups"][2], meta["duplicate_groups"][3])
        self.assertEqual(meta["team_groups"][2], meta["team_groups"][3])

    def test_duplicate_ids_survive_output_and_pairing(self):
        train = copy.deepcopy(self.data["train_records"])
        train.extend([copy.deepcopy(train[0]), copy.deepcopy(train[1])])
        data = build_record_features(train, self.data["test_records"], self.data["chart"])
        audit, meta = audit_battles(train)
        data.update(train_records=train, chart=self.data["chart"], audit=audit, audit_metadata=meta)
        with patch("pokemon.main.make_model", side_effect=TinyEstimator):
            report, oof = cross_validate(data, return_details=True)
        self.assertEqual(len(oof), len(train))
        self.assertTrue(oof["battle_id"].duplicated().any())
        self.assertTrue((oof.groupby("duplicate_group")["fold"].nunique() == 1).all())
        self.assertIn("duplicate_split_effect", report)
        comparison = compare_oof(oof, oof.iloc[::-1])
        self.assertTrue(all(m["corrected"] == m["regressed"] == 0 for m in comparison.values()))

    def test_grouped_encoder_excludes_same_p1_group_and_validation_labels(self):
        k1 = [frozenset([str(i // 2)]) for i in range(24)]
        k2 = [None] * 24
        y = np.arange(24) % 2
        groups = np.arange(24) // 2
        keys = list(dict.fromkeys(k1))
        candidates = [np.array([], int)] * 24
        fit, valid = np.arange(20), np.arange(20, 24)
        encoded, applied = oof_te2(k1, k2, y, keys, candidates, fit, valid, groups=groups)
        np.testing.assert_array_equal(encoded["te_p1_n"], 0)
        y[valid] = 1 - y[valid]
        again, again_applied = oof_te2(k1, k2, y, keys, candidates, fit, valid, groups=groups)
        pd.testing.assert_frame_equal(encoded, again)
        pd.testing.assert_frame_equal(applied, again_applied)

    def test_grouped_reports_p2_overlap_and_subgroup_counts(self):
        with patch("pokemon.main.make_model", side_effect=TinyEstimator):
            report, oof = cross_validate(self.data, split_mode="p1_grouped", return_details=True)
        self.assertEqual(int(oof["p1_seen_in_training"].sum()), 0)
        self.assertGreater(int(oof["p1_seen_as_p2_in_training"].sum()), 0)
        self.assertEqual(report["composition_subgroups"]["p1_absent_as_p1"]["n"], 30)
        for metrics in report["models"].values():
            self.assertEqual(metrics["correct"] + metrics["wrong"], 30)
            self.assertEqual(sum(f["correct"] for f in metrics["fold_counts"].values()), metrics["correct"])
            paired = metrics["vs_reference"]
            self.assertEqual(paired["corrected"] - paired["regressed"], paired["net_correct"])
        groups = protocol_groups(self.data, "p1_grouped")
        for _, rows in oof.groupby("fold"):
            valid = rows["row_id"].to_numpy()
            fit = np.setdiff1d(np.arange(30), valid)
            self.assertFalse(set(groups[fit]) & set(groups[valid]))

    def test_fold_transforms_ignore_apply_catalog_and_labels(self):
        train = self.data["train_records"][:20]
        valid = self.data["test_records"]
        first = build_record_features(train, valid, self.data["chart"], training_only=True)
        changed = copy.deepcopy(valid)
        changed[0]["player_won"] = 1 - changed[0]["player_won"]
        changed[0]["p1_team_details"][0]["base_hp"] = 999
        for t in changed[0]["battle_timeline"]:
            t["p1_move_details"]["name"] = "heldout_only"
        second = build_record_features(train, changed * 120, self.data["chart"], training_only=True)
        pd.testing.assert_frame_equal(first["A"], second["A"])
        self.assertEqual(first["preparation"], second["preparation"])
        self.assertNotIn("heldout_only", second["preparation"]["moves"])
        self.assertNotIn("team100_0", second["preparation"]["dex_species"])
        self.assertTrue(all(len(c) == 0 for c in first["C"][len(train):]))
        self.assertEqual(list(first["A"]), list(first["B"]))
        self.assertEqual(first["B"]["x_alive_a1"].iloc[0], 6)
        # A held-out label cannot affect either encoded matrix at fixed indices.
        for scope in ("global", "fold"):
            before = _validation_matrices(self.data, np.arange(20), np.arange(20, 30),
                                          preparation=scope, groups=None, te_seed=0)
            changed_data = copy.deepcopy(self.data)
            changed_data["y"][20:] = 1 - changed_data["y"][20:]
            for b in changed_data["train_records"][20:]:
                b["player_won"] = 1 - b["player_won"]
            after = _validation_matrices(changed_data, np.arange(20), np.arange(20, 30),
                                         preparation=scope, groups=None, te_seed=0)
            pd.testing.assert_frame_equal(before[0], after[0])
            pd.testing.assert_frame_equal(before[1], after[1])

    def test_nested_selection_uses_only_outer_training_and_refits_transforms(self):
        calls = []
        original_builder = build_record_features

        def spy(train, apply, chart, **kwargs):
            self.assertTrue(all("player_won" not in b for b in apply))
            fit_ids, apply_ids = {b["battle_id"] for b in train}, {b["battle_id"] for b in apply}
            self.assertFalse(fit_ids & apply_ids)
            calls.append((fit_ids, apply_ids))
            return original_builder(train, apply, chart, **kwargs)

        with tempfile.TemporaryDirectory() as folder, \
                patch("pokemon.main.make_model", side_effect=TinyEstimator), \
                patch("pokemon.main.build_record_features", side_effect=spy):
            report, oof = cross_validate(self.data, split_mode="p1_grouped", preparation="fold",
                                         nested=True, selection_splits=2, output_dir=folder,
                                         return_details=True)
            self.assertEqual(len(calls), 5 * (2 + 1))
            for choice in report["nested_selection"]["selections"]:
                fold = choice["outer_fold"]
                outer_ids = set(oof.loc[oof["fold"] == fold, "row_id"])
                inner = pd.read_csv(Path(folder) / "selection" / f"outer_{fold}" / "validation_predictions.csv")
                self.assertFalse(outer_ids & set(inner["row_id"]))
                self.assertEqual(set(inner["row_id"]), set(choice["selection_training_row_ids"]))
                winner = max(VARIANTS, key=lambda name: choice["inner_accuracies"][name])
                self.assertEqual(choice["selected"], winner)
                np.testing.assert_allclose(oof.loc[oof["fold"] == fold, "nested_selected"],
                                           oof.loc[oof["fold"] == fold, winner])
                # Builder call order is two inner fits followed by outer refit.
                outer_train = calls[fold * 3 + 2][0]
                for inner_train, inner_valid in calls[fold * 3:fold * 3 + 2]:
                    self.assertEqual(inner_train | inner_valid, outer_train)

    def test_outer_labels_do_not_change_inner_selection_at_fixed_boundary(self):
        indices = np.arange(len(self.data["y"]))
        fixed_outer = split_indices(self.data["y"], indices)
        heldout = fixed_outer[0][1]
        changed = copy.deepcopy(self.data)
        changed["y"][heldout] = 1 - changed["y"][heldout]
        for i in heldout:
            changed["train_records"][i]["player_won"] = int(changed["y"][i])

        def fixed_split(y, rows, **kwargs):
            return fixed_outer if len(rows) == len(indices) else split_indices(y, rows, **kwargs)

        with patch("pokemon.main.make_model", side_effect=TinyEstimator), \
                patch("pokemon.main.split_indices", side_effect=fixed_split):
            before, before_oof = cross_validate(self.data, nested=True, return_details=True)
            after, after_oof = cross_validate(changed, nested=True, return_details=True)
        self.assertEqual(before["nested_selection"]["selections"][0],
                         after["nested_selection"]["selections"][0])
        np.testing.assert_allclose(before_oof.loc[heldout, "nested_selected"],
                                   after_oof.loc[heldout, "nested_selected"])

    def test_preparation_comparison_uses_identical_folds(self):
        with tempfile.TemporaryDirectory() as folder, patch("pokemon.main.make_model", side_effect=TinyEstimator):
            report = compare_preparation(self.data, folder)
            a = pd.read_csv(Path(folder) / "global/validation_predictions.csv")
            b = pd.read_csv(Path(folder) / "fold/validation_predictions.csv")
            np.testing.assert_array_equal(a["fold"], b["fold"])
            compare_oof(a, b)
            self.assertIn("accuracy_delta_fold_minus_global", report["models"]["sub1_lgbm"])


if __name__ == "__main__":
    unittest.main()
