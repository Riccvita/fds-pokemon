"""Reproduce the classical transition- and forecast-model experiment.

This independent research entry point evaluates two auxiliary self-supervised
learners. All fitted auxiliary models use only observed battle states from
outer-fold training rows. Team-strength encodings use inner cross-validation.

Run
---
python transition_experiment.py --data /path/to/competition_data

The original competition submission workflow is unchanged in ``run.py``.
"""

import argparse
import json
from pathlib import Path

import joblib

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from pokemon.features import load_type_chart, read_jsonl
from pokemon.features2 import build_dex
from pokemon.forecast import build_forecast_inputs, fit_forecast_models, forecast_features
from pokemon.main import build_all, make_model
from pokemon.te import oof_te2
from pokemon.transitions import build_transition_inputs, train_transition_model, transition_features


RANDOM_SEED = 42
CANDIDATES = ("forecast_xgb", "forecast_transition_xgb")
BLEND_WEIGHTS = (0.15, 0.30, 0.50)


def prepare_inputs(data_dir):
    """Build baseline and self-supervised feature tables from official data.

    Parameters
    ----------
    data_dir : pathlib.Path
        Folder with train/test JSONL files and the official type chart.

    Returns
    -------
    tuple
        Baseline matrix dictionary, learned-transition inputs, and
        forecast inputs, all aligned by battle index.
    """
    baseline = build_all(data_dir)

    # --- Species attributes are read from the competition input itself.
    train = read_jsonl(data_dir / "train.jsonl")
    test = read_jsonl(data_dir / "test.jsonl")
    dex = build_dex(train, test)
    chart = load_type_chart(data_dir / "type_chart_gen2.csv")
    battles = train + test

    transitions = build_transition_inputs(battles, dex, chart)
    forecasts = build_forecast_inputs(battles, dex)

    return baseline, transitions, forecasts


def evaluate(baseline, transitions, forecasts, include_original=True):
    """Perform five outer splits without training on their validation labels.

    Parameters
    ----------
    baseline : dict
        Numeric features and unsupervised team candidates from ``build_all``.
    transitions : TransitionInputs
        Move/health transition data with source-battle indices.
    forecasts : ForecastInputs
        Early-turn ten-step resource changes with source-battle indices.
    include_original : bool
        Whether to retrain all original models and evaluate their blends.

    Returns
    -------
    dict
        Full-fold accuracies and AUCs for each candidate, with ensemble
        comparisons if ``include_original`` is True.
    """
    y, features = baseline["y"], baseline["A"]
    names = list(CANDIDATES)
    if include_original:
        names.extend(("lgb", "xgb", "cat"))

    predicted = {name: np.zeros(len(y), dtype=np.float64) for name in names}
    splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    for fold, (train_idx, validation_idx) in enumerate(splits.split(features, y), start=1):
        print(f"Outer fold {fold}/5", flush=True)

        # --- Fit each auxiliary learner on ONLY this outer training fold.
        damage_model = train_transition_model(transitions, train_idx)
        forecast_models = fit_forecast_models(forecasts, train_idx)
        damage_train = transition_features(damage_model, transitions, train_idx)
        damage_val = transition_features(damage_model, transitions, validation_idx)
        forecast_train = forecast_features(forecast_models, forecasts, train_idx)
        forecast_val = forecast_features(forecast_models, forecasts, validation_idx)

        # --- Supervised target encodings also exclude each validation fold.
        encoded_train, encoded_val = oof_te2(
            baseline["k1"], baseline["k2"], y,
            baseline["keys"], baseline["C"], train_idx, validation_idx,
        )
        train_base = pd.concat([
            features.iloc[train_idx].reset_index(drop=True), encoded_train,
        ], axis=1)
        val_base = pd.concat([
            features.iloc[validation_idx].reset_index(drop=True), encoded_val,
        ], axis=1)

        # --- Compare forecast-only against forecast + transition features.
        for name in CANDIDATES:
            extra_train, extra_val = forecast_train, forecast_val
            if name == "forecast_transition_xgb":
                extra_train = pd.concat([forecast_train, damage_train], axis=1)
                extra_val = pd.concat([forecast_val, damage_val], axis=1)

            estimator = make_model("xgb")
            estimator.fit(pd.concat([train_base, extra_train], axis=1), y[train_idx])
            predicted[name][validation_idx] = estimator.predict_proba(
                pd.concat([val_base, extra_val], axis=1)
            )[:, 1]

        if include_original:
            for name in ("lgb", "xgb", "cat"):
                estimator = make_model(name)
                estimator.fit(train_base, y[train_idx])
                predicted[name][validation_idx] = estimator.predict_proba(val_base)[:, 1]

        for name in CANDIDATES:
            fold_accuracy = accuracy_score(y[validation_idx], predicted[name][validation_idx] > 0.5)
            print(f"    {name}: {fold_accuracy:.2%}", flush=True)

    # --- Report classification accuracy (primary metric) and ROC AUC (secondary).
    scores = {}
    for name, prob in predicted.items():
        scores[name] = {
            "accuracy": float(accuracy_score(y, prob > 0.5)),
            "roc_auc": float(roc_auc_score(y, prob)),
        }

    if include_original:
        reference = (0.45 * predicted["lgb"] + 0.45 * predicted["xgb"]
                     + 0.10 * predicted["cat"])
        scores["original_ensemble"] = {
            "accuracy": float(accuracy_score(y, reference > 0.5)),
            "roc_auc": float(roc_auc_score(y, reference)),
        }
        for name in CANDIDATES:
            for weight in BLEND_WEIGHTS:
                blended = (1-weight) * reference + weight * predicted[name]
                scores[f"original_plus_{name}_{weight:.2f}"] = {
                    "accuracy": float(accuracy_score(y, blended > 0.5)),
                    "roc_auc": float(roc_auc_score(y, blended)),
                }

    return scores


def main():
    """Parse arguments, evaluate the experiment, and save its score report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="Directory with the official competition inputs")
    parser.add_argument("--out", type=Path, default=Path("docs/transition_cv.json"),
                        help="Where to write five-fold accuracy and AUC results")
    parser.add_argument("--skip-original", action="store_true",
                        help="Skip refitting the original three base models")
    parser.add_argument("--baseline-cache", type=Path,
                        help="Optional precomputed baseline feature dictionary (joblib)")
    parser.add_argument("--transition-cache", type=Path,
                        help="Optional precomputed transition tables (joblib)")
    parser.add_argument("--forecast-cache", type=Path,
                        help="Optional precomputed forecast tables (joblib)")
    args = parser.parse_args()

    # --- Load and validate input paths before starting expensive calculations.
    for name in ("train.jsonl", "test.jsonl", "type_chart_gen2.csv"):
        if not (args.data / name).is_file():
            parser.error(f"Missing competition input: {args.data / name}")

    # --- Reuse caches when available; otherwise build from official files.
    if args.baseline_cache and args.transition_cache and args.forecast_cache:
        baseline = joblib.load(args.baseline_cache)
        transitions = joblib.load(args.transition_cache)
        forecasts = joblib.load(args.forecast_cache)
    else:
        baseline, transitions, forecasts = prepare_inputs(args.data)

    scores = evaluate(baseline, transitions, forecasts, include_original=not args.skip_original)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
