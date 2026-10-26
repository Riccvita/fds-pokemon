"""Behavioral checks for saved OOF predictions and paired comparisons."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from pokemon.main import MODEL_NAMES, SEED, cross_validate
from pokemon.validation import compare_oof, summarize_validation


class ValidationReportTests(unittest.TestCase):
    def table(self):
        # A net gain of one hides two corrections and one regression.
        return pd.DataFrame({
            "battle_id": ["001", "002", "003", "004", "005"],
            "fold": [0, 1, 2, 3, 4],
            "player_won": [0, 1, 0, 1, 0],
            "sub1_lgbm": [0.9, 0.2, 0.2, 0.8, 0.7],
            "candidate": [0.5, 0.8, 0.9, 0.8, 0.7],
        })

    def test_gross_changes_and_fold_scores(self):
        report = summarize_validation(self.table())
        candidate = report["models"]["candidate"]
        self.assertEqual(candidate["accuracy"], 3 / 5)
        self.assertEqual(candidate["fold_accuracies"],
                         {"0": 1., "1": 1., "2": 0., "3": 1., "4": 0.})
        self.assertEqual(candidate["vs_reference"], {
            "corrected": 2, "regressed": 1, "net_correct": 1,
            "both_correct": 1, "both_wrong": 1,
        })

    def test_previous_run_is_aligned_by_id(self):
        previous = self.table().iloc[::-1].copy()
        previous["candidate"] = previous["sub1_lgbm"]
        result = compare_oof(self.table(), previous)
        self.assertEqual(result["candidate"]["corrected"], 2)
        self.assertEqual(result["candidate"]["regressed"], 1)
        self.assertEqual(result["sub1_lgbm"]["net_correct"], 0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "previous.csv.gz"
            previous.to_csv(path, index=False)
            report = summarize_validation(self.table(), reference_oof=path)
        self.assertEqual(report["previous_run"]["models"], result)

    def test_saved_probabilities_preserve_threshold_boundary(self):
        table = self.table()
        table.loc[0, "candidate"] = np.nextafter(0.5, 1.0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "previous.csv.gz"
            table.to_csv(path, index=False)
            report = summarize_validation(table, reference_oof=path)
        comparison = report["previous_run"]["models"]["candidate"]
        self.assertEqual(comparison["corrected"], 0)
        self.assertEqual(comparison["regressed"], 0)

    def test_incompatible_runs_are_rejected(self):
        table = self.table()
        for column, value in (("battle_id", "other"), ("fold", 4),
                              ("player_won", 1), ("candidate", np.nan),
                              ("candidate", 1.1)):
            with self.subTest(column=column, value=value):
                previous = table.copy()
                previous.loc[0, column] = value
                with self.assertRaises(ValueError):
                    compare_oof(table, previous)
        for previous in (table.iloc[:-1], pd.concat([table, table.iloc[:1]])):
            with self.assertRaises(ValueError):
                compare_oof(table, previous)


class CrossValidationTests(unittest.TestCase):
    @staticmethod
    def data():
        # Global majority is class 1. Some outer training folds are tied,
        # so a global majority shortcut would produce incorrect predictions.
        return {
            "A": pd.DataFrame({"row": np.arange(11)}, index=np.arange(20, 31)),
            "y": np.array([0] * 5 + [1] * 6),
            "train_ids": [f"{i:03d}" for i in reversed(range(11))],
            "k1": [], "k2": [], "keys": [], "C": [],
        }

    def run_cv(self, data, **kwargs):
        class Estimator:
            def get_params(self):
                return {}

            def fit(self, X, y):
                self.training_rows = set(X["row"])

            def predict_proba(self, X):
                if self.training_rows.intersection(X["row"]):
                    raise AssertionError("Prediction includes a training battle")
                p = X["row"].to_numpy() / 10
                return np.column_stack([1 - p, p])

        def encoder(k1, k2, y, keys, candidates, train_idx, valid_idx, **kwargs):
            return pd.DataFrame(index=range(len(train_idx))), pd.DataFrame(index=range(len(valid_idx)))

        with patch("pokemon.main.make_model", side_effect=lambda name, **kwargs: Estimator()), \
                patch("pokemon.main.oof_te2", side_effect=encoder):
            return cross_validate(data, **kwargs)

    def test_oof_alignment_training_majority_and_saved_artifacts(self):
        data = self.data()
        with tempfile.TemporaryDirectory() as folder:
            report, oof = self.run_cv(data, output_dir=folder, return_details=True)
            saved = pd.read_csv(Path(folder) / "validation_predictions.csv", dtype={"battle_id": str})
            self.assertEqual(json.loads((Path(folder) / "validation_report.json").read_text()), report)
            pd.testing.assert_frame_equal(saved, oof)
            # Reusing the artifact compares the same held-out predictions.
            repeated, _ = self.run_cv(data, reference_oof=Path(folder) / "validation_predictions.csv",
                                      return_details=True)
            self.assertTrue(all(metrics["corrected"] == metrics["regressed"] == 0
                                for metrics in repeated["previous_run"]["models"].values()))
        self.assertEqual(list(oof["battle_id"]), data["train_ids"])
        np.testing.assert_array_equal(oof["player_won"], data["y"])
        for name in MODEL_NAMES:
            np.testing.assert_allclose(oof[name], np.arange(11) / 10)
        for fold, (train_idx, valid_idx) in enumerate(
            StratifiedKFold(5, shuffle=True, random_state=SEED).split(data["A"], data["y"])
        ):
            np.testing.assert_array_equal(oof.iloc[valid_idx]["fold"], fold)
            expected = int(data["y"][train_idx].mean() > 0.5)
            np.testing.assert_array_equal(oof.iloc[valid_idx]["majority"], expected)
        self.assertEqual(set(oof["majority"]), {0., 1.})
        self.assertEqual(len(report["fold_sizes"]), 5)
        scores = self.run_cv(data)
        self.assertEqual(scores, {name: metrics["accuracy"]
                                  for name, metrics in report["models"].items()})


if __name__ == "__main__":
    unittest.main()
