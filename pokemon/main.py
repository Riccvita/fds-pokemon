"""Training, leakage-safe validation, and Kaggle submission generation.

All supervised estimators are trained exclusively on ``train.jsonl`` labels.
Team target statistics are cross-fitted within each training fold to prevent
outcome leakage into either training features or validation predictions.

The submission variants were selected using stratified five-fold validation:

    sub1_lgbm:        LightGBM alone.
    sub2_lgb_xgb:     Equal-probability blend of LightGBM and XGBoost.
    sub3_lgb_xgb_cat: LightGBM/XGBoost/CatBoost blend (45%/45%/10%).

Do not use the competition test outcomes or public leaderboard to tune features.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from pokemon.features import build_frame, load_type_chart, read_jsonl
from pokemon.features2 import build_dex, build_extras
from pokemon.features3 import build3
from pokemon.features4 import cand_lists, feats4, pool_matrix
from pokemon.te import oof_te2, team_keys


SEED = 42
MODEL_NAMES = ("lgb", "xgb", "cat")

# Positive weights are probability averaging coefficients, not voting weights.
VARIANTS = {
    "sub1_lgbm": {"lgb": 1.0},
    "sub2_lgb_xgb": {"lgb": 0.50, "xgb": 0.50},
    "sub3_lgb_xgb_cat": {"lgb": 0.45, "xgb": 0.45, "cat": 0.10},
}


def build_all(data_dir):
    """Build numerical training/test matrices from competition-provided inputs.

    Parameters
    ----------
    data_dir : str or pathlib.Path
        Directory containing ``train.jsonl``, ``test.jsonl`` and
        ``type_chart_gen2.csv``.

    Returns
    -------
    dict
        ``A`` and ``B`` are matching unlabeled feature matrices; ``y`` contains
        training outcomes, ``k1``/``k2`` are team identifiers, ``keys`` and ``C``
        describe candidate teams, and ``ids`` preserves the original test IDs.

    Notes
    -----
    The vocabulary of move and species features is defined from train only.
    The species base-stat dictionary comes exclusively from supplied records.
    Candidate sets only inspect team identities, never prediction targets.
    """
    data_dir = Path(data_dir)

    # --- Read supplied files without any external Pokémon knowledge base.
    chart = load_type_chart(data_dir / "type_chart_gen2.csv")
    train = read_jsonl(data_dir / "train.jsonl")
    test = read_jsonl(data_dir / "test.jsonl")
    y = np.asarray([battle["player_won"] for battle in train], dtype=int)

    dex = build_dex(train, test)

    # --- Build reusable, entirely label-independent numerical feature groups.
    base_train, moves, species = build_frame(train, chart)
    base_test, _, _ = build_frame(test, chart, moves, species)

    extras_train, slot_columns = build_extras(train, dex)
    extras_test, _ = build_extras(test, dex, slot_columns)

    knockouts_train = build3(train)
    knockouts_test = build3(test)

    # --- Infer candidate P2 lineups from unlabeled observed team identities.
    # The pool matches the original reference pipeline. The experimental
    # two-sided pool was tested separately and did not improve held-out accuracy.
    keys, membership, weights, lookup, species_names = pool_matrix(train + test, dex)
    candidates = cand_lists(train + test, membership, lookup)

    pool_features = pd.DataFrame([
        feats4(battle, candidates[i], membership, weights, species_names, dex, chart)
        for i, battle in enumerate(train + test)
    ])

    n_train = len(train)
    pool_train = pool_features.iloc[:n_train].reset_index(drop=True)
    pool_test = pool_features.iloc[n_train:].reset_index(drop=True)

    # --- Maintain exact train/test column correspondence.
    A = pd.concat([
        base_train.drop(columns="battle_id"), extras_train,
        knockouts_train, pool_train,
    ], axis=1)
    B = pd.concat([
        base_test.drop(columns="battle_id"), extras_test,
        knockouts_test, pool_test,
    ], axis=1).reindex(columns=A.columns)

    k1, k2 = team_keys(train + test)

    return {
        "A": A,
        "B": B,
        "y": y,
        "k1": k1,
        "k2": k2,
        "keys": keys,
        "C": candidates,
        "ids": [battle["battle_id"] for battle in test],
    }


def make_model(name, seed=SEED):
    """Instantiate a seeded classical gradient-boosting classifier.

    Parameters
    ----------
    name : {"lgb", "xgb", "cat"}
        Identifier of the modeling algorithm.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    sklearn-compatible classifier
        A fresh unfitted estimator with ``predict_proba`` support.
    """
    if name == "lgb":
        return lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.03,
            num_leaves=8,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.4,
            min_child_samples=40,
            reg_lambda=10,
            verbose=-1,
            random_state=seed,
            n_jobs=2,
        )

    if name == "xgb":
        return XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.025,
            colsample_bytree=0.75,
            subsample=0.85,
            min_child_weight=12,
            reg_lambda=20,
            tree_method="hist",
            random_state=seed,
            n_jobs=2,
        )

    if name == "cat":
        return CatBoostClassifier(
            iterations=500,
            depth=5,
            learning_rate=0.04,
            l2_leaf_reg=8,
            rsm=0.3,
            verbose=False,
            random_seed=seed,
            thread_count=2,
            allow_writing_files=False,
        )

    raise ValueError(f"Unknown model: {name!r}. Expected one of {MODEL_NAMES}.")


def blend_probabilities(predictions, weights):
    """Compute a normalized weighted average of calibrated class-1 probabilities.

    Parameters
    ----------
    predictions : dict[str, numpy.ndarray]
        Predicted class-1 probabilities, keyed by model name.
    weights : dict[str, float]
        Non-negative model coefficients that sum to one.

    Returns
    -------
    numpy.ndarray
        Ensemble class-1 probabilities with the original row order preserved.
    """
    if not weights or not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("Ensemble weights must sum to 1.0.")

    return sum(weight * predictions[name] for name, weight in weights.items())


def cross_validate(data, n_splits=5):
    """Estimate held-out accuracy with strictly nested team target encoding.

    Each of the ``n_splits`` validation folds is never used to fit that fold's
    predictors or to compute its target-encoding statistics. The inner K-fold
    encoder also excludes the outcome of each row it encodes for training.

    Returns
    -------
    dict[str, float]
        Out-of-fold accuracies of base estimators and submission variants.
    """
    A, y = data["A"], data["y"]
    n_train = len(y)

    predictions = {name: np.zeros(n_train) for name in MODEL_NAMES}
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for train_idx, validation_idx in outer_cv.split(A, y):
        # --- Fit supervised target encoders exclusively on outer train rows.
        enc_train, enc_validation = oof_te2(
            data["k1"], data["k2"], y, data["keys"], data["C"],
            train_idx, validation_idx,
        )

        X_train = pd.concat([
            A.iloc[train_idx].reset_index(drop=True), enc_train,
        ], axis=1)
        X_validation = pd.concat([
            A.iloc[validation_idx].reset_index(drop=True), enc_validation,
        ], axis=1)

        # --- Save probabilities only for the untouched outer validation rows.
        for name in MODEL_NAMES:
            estimator = make_model(name)
            estimator.fit(X_train, y[train_idx])
            predictions[name][validation_idx] = estimator.predict_proba(X_validation)[:, 1]

    scores = {
        name: float(np.mean((probabilities > 0.5) == y))
        for name, probabilities in predictions.items()
    }

    for name, weights in VARIANTS.items():
        probabilities = blend_probabilities(predictions, weights)
        scores[name] = float(np.mean((probabilities > 0.5) == y))

    return scores


def _final_matrices(data):
    """Create full-train and test matrices with non-leaking target encodings."""
    A, B, y = data["A"], data["B"], data["y"]
    n_train = len(y)

    # --- Training rows use inner out-of-fold values; test rows use all labels.
    enc_train, enc_test = oof_te2(
        data["k1"], data["k2"], y, data["keys"], data["C"],
        np.arange(n_train), np.arange(n_train, n_train + len(B)),
    )

    X_train = pd.concat([A.reset_index(drop=True), enc_train], axis=1)
    X_test = pd.concat([B.reset_index(drop=True), enc_test], axis=1)

    return X_train, X_test


def _write_submission(ids, probabilities, output_path):
    """Write binary predictions while retaining their original battle IDs."""
    submission = pd.DataFrame({
        "battle_id": ids,
        "player_won": (np.asarray(probabilities) > 0.5).astype(int),
    })
    submission.to_csv(output_path, index=False)


def make_submission(data, name, out_dir):
    """Fit only the estimators needed for one selected Kaggle submission."""
    if name not in VARIANTS:
        raise ValueError(f"Unknown submission: {name!r}.")

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_test = _final_matrices(data)

    # --- Generate each base estimator's probability exactly once.
    probabilities = {}
    for model_name in VARIANTS[name]:
        estimator = make_model(model_name)
        estimator.fit(X_train, data["y"])
        probabilities[model_name] = estimator.predict_proba(X_test)[:, 1]

    output_path = output_dir / f"{name}.csv"
    _write_submission(
        data["ids"], blend_probabilities(probabilities, VARIANTS[name]), output_path,
    )

    return str(output_path)


def make_submissions(data, out_dir):
    """Train each estimator once and regenerate all three final submissions.

    Returns
    -------
    dict[str, str]
        Mapping from the submission variant to its output CSV path.
    """
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_test = _final_matrices(data)

    # --- Reuse identical fitted models across the three ensemble variants.
    probabilities = {}
    for name in MODEL_NAMES:
        estimator = make_model(name)
        estimator.fit(X_train, data["y"])
        probabilities[name] = estimator.predict_proba(X_test)[:, 1]

    paths = {}
    for variant, weights in VARIANTS.items():
        path = output_dir / f"{variant}.csv"
        _write_submission(data["ids"], blend_probabilities(probabilities, weights), path)
        paths[variant] = str(path)

    return paths
