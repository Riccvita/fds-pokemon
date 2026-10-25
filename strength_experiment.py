"""Validate active-strength features and seed averaging on the original folds.

Example:
    python strength_experiment.py --data /path/to/competition --submission

The primary candidate averages five XGBoost models. A second, fixed candidate
averages that probability with the original 45/45/10 ensemble. All decision
thresholds are 0.5; no validation labels are used to fit models or encodings.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from pokemon.main import (
    MODEL_NAMES, SEED, VARIANTS, _final_matrices, _write_submission,
    blend_probabilities, build_all, make_model,
)
from pokemon.strength_features import build_strength_features
from pokemon.te import oof_te2


SEEDS = (42, 137, 2026, 31415, 271828)
REFERENCE_WEIGHTS = VARIANTS["sub3_lgb_xgb_cat"]
PROFILES = {
    "default": {},
    "longer": {"n_estimators": 800, "learning_rate": 0.02},
    "shallow": {"n_estimators": 800, "max_depth": 3},
    "deeper": {"n_estimators": 600, "max_depth": 5, "min_child_weight": 20, "reg_lambda": 30},
    "regularized": {"colsample_bytree": 0.5, "subsample": 1.0,
                    "min_child_weight": 20, "reg_lambda": 30},
}


def make_strength_model(seed=SEED, profile="default"):
    """Instantiate one of the explicitly documented experimental profiles."""
    return make_model("xgb", seed=seed).set_params(**PROFILES[profile])


def _augment(frame):
    """Add deterministic interactions to a matrix without changing its rows."""
    return pd.concat([frame, build_strength_features(frame)], axis=1)


def summarize_predictions(y, folds, reference, strength_by_seed, seeds=SEEDS):
    """Score fixed candidates and record all seeds, rather than selecting a seed."""
    strength = strength_by_seed.mean(axis=0)
    candidates = {
        "reference": reference,
        "strength_xgb": strength,
        "strength_reference_blend": 0.5 * (reference + strength),
    }
    results = {}
    original_correct = (reference > 0.5) == y
    for name, probabilities in candidates.items():
        correct = (probabilities > 0.5) == y
        results[name] = {
            "accuracy": float(accuracy_score(y, probabilities > 0.5)),
            "roc_auc": float(roc_auc_score(y, probabilities)),
            "correct": int(correct.sum()),
            "fold_accuracies": [
                float(correct[folds == fold].mean()) for fold in sorted(set(folds))
            ],
            "corrected_vs_reference": int((correct & ~original_correct).sum()),
            "regressed_vs_reference": int((~correct & original_correct).sum()),
        }
    return {
        "n_observations": len(y),
        "outer_fold_seed": SEED,
        "seeds": list(seeds),
        "threshold": 0.5,
        "n_strength_features": 42,
        "test_labels_used": False,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "catboost")
        },
        "candidates": results,
        "individual_strength_seed_accuracies": {
            str(seed): float(accuracy_score(y, probabilities > 0.5))
            for seed, probabilities in zip(seeds, strength_by_seed)
        },
    }, candidates


def validate_strength(data, seeds=SEEDS, profile="default"):
    """Compare the original ensemble and the fixed candidates using nested CV."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide at least one seed, with no duplicates.")
    A, y = data["A"], data["y"]
    baseline = {name: np.full(len(y), np.nan) for name in MODEL_NAMES}
    strength = np.full((len(seeds), len(y)), np.nan)
    folds = np.full(len(y), -1, dtype=int)

    for fold, (train_idx, valid_idx) in enumerate(
        StratifiedKFold(5, shuffle=True, random_state=SEED).split(A, y),
    ):
        enc_train, enc_valid = oof_te2(
            data["k1"], data["k2"], y, data["keys"], data["C"],
            train_idx, valid_idx,
        )
        train = pd.concat([A.iloc[train_idx].reset_index(drop=True), enc_train], axis=1)
        valid = pd.concat([A.iloc[valid_idx].reset_index(drop=True), enc_valid], axis=1)
        folds[valid_idx] = fold

        for name in MODEL_NAMES:
            model = make_model(name)
            model.fit(train, y[train_idx])
            baseline[name][valid_idx] = model.predict_proba(valid)[:, 1]

        strength_train, strength_valid = _augment(train), _augment(valid)
        for position, seed in enumerate(seeds):
            model = make_strength_model(seed=seed, profile=profile)
            model.fit(strength_train, y[train_idx])
            strength[position, valid_idx] = model.predict_proba(strength_valid)[:, 1]

        probabilities = strength[:, valid_idx].mean(axis=0)
        score = accuracy_score(y[valid_idx], probabilities > 0.5)
        print(f"Fold {fold + 1}: strength XGBoost {score:.4%}", flush=True)

    if not np.isfinite(strength).all() or (folds < 0).any():
        raise RuntimeError("Every labeled row must receive exactly one OOF prediction.")
    reference = blend_probabilities(baseline, REFERENCE_WEIGHTS)
    results, candidates = summarize_predictions(y, folds, reference, strength, seeds)
    results["profile"] = profile
    results["profile_overrides"] = PROFILES[profile]
    results["baseline_model_accuracies"] = {
        name: float(accuracy_score(y, probabilities > 0.5))
        for name, probabilities in baseline.items()
    }
    oof = pd.DataFrame({"fold": folds, "player_won": y, **candidates})
    for seed, probabilities in zip(seeds, strength):
        oof[f"strength_seed_{seed}"] = probabilities
    return results, oof


def generate_candidates(data, output_dir, seeds=SEEDS, profile="default"):
    """Refit the same fixed candidates on all labels and preserve test ID order."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, test = _final_matrices(data)
    base_probabilities = {}
    for name in MODEL_NAMES:
        model = make_model(name)
        model.fit(train, data["y"])
        base_probabilities[name] = model.predict_proba(test)[:, 1]
    reference = blend_probabilities(base_probabilities, REFERENCE_WEIGHTS)
    train, test = _augment(train), _augment(test)
    strength = np.zeros(len(test))
    for seed in seeds:
        model = make_strength_model(seed=seed, profile=profile)
        model.fit(train, data["y"])
        strength += model.predict_proba(test)[:, 1] / len(seeds)
    paths = {}
    for name, probabilities in {
        "candidate_strength_xgb": strength,
        "candidate_strength_blend": 0.5 * (reference + strength),
    }.items():
        path = output_dir / f"{name}.csv"
        _write_submission(data["ids"], probabilities, path)
        paths[name] = str(path)
    return paths


def main():
    """Save reproducible scores and OOF probabilities, optionally candidate CSVs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("docs/STRENGTH_CV_RESULTS.json"))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--profile", choices=PROFILES, default="default")
    parser.add_argument("--submission", action="store_true")
    parser.add_argument("--submission-dir", type=Path, default=Path("submissions/candidates"))
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")

    data = build_all(args.data)
    results, oof = validate_strength(data, seeds=args.seeds, profile=args.profile)
    # Bind positional OOF outputs to the exact source file used in this run.
    with (args.data / "train.jsonl").open("rb") as source:
        results["train_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    oof_path = args.out.with_suffix(".oof.csv.gz")
    oof.to_csv(oof_path, index=False)
    results["oof_file"] = oof_path.name
    if args.submission:
        results["submission_files"] = generate_candidates(
            data, args.submission_dir, args.seeds, args.profile,
        )
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results["candidates"], indent=2))


if __name__ == "__main__":
    main()
