"""Evaluate leakage-safe next-action features on the Pokémon battle challenge.

Usage
-----
python action_experiment.py --data /path/to/input --baseline-cache /path/baseline.joblib \
    --reference-oof /path/all_oof.joblib --folds 5

The action model is trained *only* on self-supervised next-move labels from the
outer training battles. Inner-fold target encodings are generated independently.
This script never uses competition test outcomes or public leaderboard results.
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from pokemon.actions import action_features, build_action_inputs, fit_action_model
from pokemon.features import load_type_chart, read_jsonl
from pokemon.features2 import build_dex
from pokemon.main import build_all, make_model
from pokemon.te import oof_te2


SEED = 42
ORIGINAL_WEIGHTS = {"lgb": 0.45, "xgb": 0.45, "cat": 0.10}


def load_inputs(args):
    """Load cached matrices or build competition-only action/benchmark tables."""
    started = time.monotonic()
    path = Path(args.data)
    baseline = joblib.load(args.baseline_cache) if args.baseline_cache else build_all(path)

    if args.action_cache and Path(args.action_cache).exists():
        actions = joblib.load(args.action_cache)
    else:
        train = read_jsonl(path / "train.jsonl")
        print("Training battles loaded; preparing causal action snapshots", flush=True)
        dex = build_dex(train)
        chart = load_type_chart(path / "type_chart_gen2.csv")

        # --- Holdout CV uses training battles only; test is not consulted.
        actions = build_action_inputs(train, dex, chart)

        if args.action_cache:
            joblib.dump(actions, args.action_cache, compress=1)

    print(f"Input build/load: {time.monotonic()-started:.1f}s", flush=True)
    return baseline, actions


def evaluate(data, actions, reference, folds=5, output=None, include_xgb=False):
    """Compare direct action features and blended probabilities using held-out folds.

    Parameters
    ----------
    data : dict
        Standard original feature matrices and team keys.
    actions : ActionInputs
        Action-classifier contexts with source battle identifiers.
    reference : dict[str, numpy.ndarray]
        Independently generated out-of-fold predictions of original models.
    folds : int
        How many consecutive folds to calculate (up to five).
    output : Path or None
        Optional checkpoint path for OOF probabilities.
    include_xgb : bool
        Whether to train the additional XGBoost outcome-model ablation.

    Returns
    -------
    dict
        Full-fold or partial-fold accuracy/AUC measurements and prediction arrays.
    """
    base = data["A"]
    y = data["y"]
    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(base, y))
    model_types = {"action_lgb": "lgb"}
    if include_xgb:
        model_types["action_xgb"] = "xgb"

    oof = {key: np.full(len(y), np.nan) for key in model_types}
    action_cv = []

    if output and output.exists():
        existing = joblib.load(output)
        oof = existing["predictions"]
        action_cv = existing["action_validation"]

    for number, (train_idx, valid_idx) in enumerate(splits[:folds], start=1):
        if all(np.isfinite(oof[name][valid_idx]).all() for name in model_types):
            print(f"Skipping previously completed fold {number}", flush=True)
            continue
        started = time.monotonic()
        print(f"Fold {number}/5", flush=True)

        # --- Fit action labels only from outer training battles.
        model = fit_action_model(actions, train_idx)
        train_aux = action_features(model, actions, train_idx)
        valid_aux = action_features(model, actions, valid_idx)

        # --- Check true out-of-fold action learning on held-out prefixes.
        mask = np.isin(actions.history_battle_index, valid_idx)
        action_true = actions.history_target[mask]
        action_prob = model.predict_proba(actions.history.loc[mask])
        action_cv.append({
            "accuracy": float(accuracy_score(action_true, np.argmax(action_prob, axis=1))),
            "log_loss": float(log_loss(action_true, action_prob, labels=[0, 1, 2])),
            "majority_baseline": float(np.max(np.bincount(action_true, minlength=3)) / len(action_true)),
        })

        # --- Match the exact original nested encoding and validation folds.
        enc_train, enc_valid = oof_te2(
            data["k1"], data["k2"], y,
            data["keys"], data["C"], train_idx, valid_idx,
        )
        X_train = pd.concat([base.iloc[train_idx].reset_index(drop=True),
                             enc_train, train_aux], axis=1)
        X_valid = pd.concat([base.iloc[valid_idx].reset_index(drop=True),
                             enc_valid, valid_aux], axis=1)

        for name, algorithm in model_types.items():
            if algorithm == "xgb":
                # --- Lightweight XGBoost ablation; no leaderboard selection.
                estimator = XGBClassifier(
                    n_estimators=220, max_depth=4, learning_rate=0.045,
                    colsample_bytree=0.8, subsample=0.85,
                    min_child_weight=12, reg_lambda=20,
                    tree_method="hist", random_state=SEED, n_jobs=3,
                )
            else:
                estimator = make_model(algorithm)
            estimator.fit(X_train, y[train_idx])
            probabilities = estimator.predict_proba(X_valid)[:, 1]
            oof[name][valid_idx] = probabilities

            print(f"  {name}: {accuracy_score(y[valid_idx], probabilities > 0.5):.4%}", flush=True)

        print(f"  Action choice accuracy: {action_cv[-1]['accuracy']:.4%} "
              f"(majority {action_cv[-1]['majority_baseline']:.4%})", flush=True)
        print(f"  Elapsed: {time.monotonic()-started:.1f}s", flush=True)

        if output:
            joblib.dump({"predictions": oof, "action_validation": action_cv}, output, compress=1)

    covered = np.isfinite(oof["action_lgb"])
    original = sum(reference[name] * weight for name, weight in ORIGINAL_WEIGHTS.items())
    scores = {"n_validation": int(covered.sum()),
              "original_ensemble_accuracy": float(accuracy_score(y[covered], original[covered] > 0.5)),
              "action_classifier": action_cv}

    for name, candidate in oof.items():
        scores[name] = {
            "accuracy": float(accuracy_score(y[covered], candidate[covered] > 0.5)),
            "auc": float(roc_auc_score(y[covered], candidate[covered])),
        }
        for weight in (0.15, 0.30, 0.50):
            blended = weight * candidate[covered] + (1.0-weight) * original[covered]
            scores[f"original_plus_{name}_{weight}"] = {
                "accuracy": float(accuracy_score(y[covered], blended > 0.5)),
                "auc": float(roc_auc_score(y[covered], blended)),
            }

    for key, result in scores.items():
        if isinstance(result, dict) and "accuracy" in result:
            print(f"{key}: {result['accuracy']:.4%} AUC={result['auc']:.5f}", flush=True)
    return scores


def compute_original_oof(data):
    """Rebuild original held-out model probabilities if no cache is provided.

    This is slower than loading validated OOF arrays, but it makes the action
    experiment fully reproducible with just the competition input files.
    """
    y = data["y"]
    predictions = {name: np.full(len(y), np.nan) for name in ORIGINAL_WEIGHTS}
    splits = StratifiedKFold(5, shuffle=True, random_state=SEED)

    for number, (train_idx, valid_idx) in enumerate(splits.split(data["A"], y), start=1):
        print(f"Rebuilding original OOF fold {number}/5", flush=True)

        encoding_train, encoding_val = oof_te2(
            data["k1"], data["k2"], y,
            data["keys"], data["C"], train_idx, valid_idx,
        )
        train_frame = pd.concat([
            data["A"].iloc[train_idx].reset_index(drop=True), encoding_train,
        ], axis=1)
        valid_frame = pd.concat([
            data["A"].iloc[valid_idx].reset_index(drop=True), encoding_val,
        ], axis=1)

        for name in predictions:
            model = make_model(name)
            model.fit(train_frame, y[train_idx])
            predictions[name][valid_idx] = model.predict_proba(valid_frame)[:, 1]

    return predictions


def main():
    """Parse paths, execute the experiment, and persist reproducible results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--baseline-cache", type=Path)
    parser.add_argument("--reference-oof", type=Path)
    parser.add_argument("--action-cache", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/ACTION_CV_RESULTS.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("docs/ACTION_OOF.joblib"))
    parser.add_argument("--folds", type=int, choices=(1, 2, 3, 4, 5), default=5)
    parser.add_argument("--include-xgb", action="store_true")
    args = parser.parse_args()

    data, actions = load_inputs(args)
    reference = (joblib.load(args.reference_oof) if args.reference_oof
                 else compute_original_oof(data))
    results = evaluate(data, actions, reference, args.folds, args.checkpoint, args.include_xgb)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
