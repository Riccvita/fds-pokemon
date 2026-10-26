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
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from pokemon.features import build_frame, load_type_chart, read_jsonl
from pokemon.features2 import build_dex, build_extras
from pokemon.features3 import build3
from pokemon.features4 import cand_lists, feats4, pool_matrix
from pokemon.te import oof_te2, team_keys
from pokemon.validation import summarize_validation, write_validation, with_decisions, model_columns
from pokemon.protocols import (REPEAT_SEEDS, audit_battles, digest, file_digest,
                               protocol_groups, provenance, split_indices)


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
        describe candidate teams, ``train_ids`` preserves training battle IDs,
        and ``ids`` preserves the original test IDs.

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
    data = build_record_features(train, test, chart)
    audit, metadata = audit_battles(train, test)
    data.update({
        "train_records": train, "test_records": test, "chart": chart,
        "audit": audit, "audit_metadata": metadata,
        "training_records_sha256": digest(train),
        "input_sha256": {name: file_digest(data_dir / name) for name in
                         ("train.jsonl", "test.jsonl", "type_chart_gen2.csv")},
    })
    return data


def build_record_features(train, test, chart, *, training_only=False):
    """Fit vocabularies on train; optionally fit dex/pool on train only.

    Apply records never contribute to fitted state in training_only mode.
    Per-battle direct observations remain usable, including supplied base stats.
    Unknown vocabulary entries have no indicator column; unknown dex entries
    use the existing NaN/default resource handling and no candidate matches.
    """
    y = np.asarray([battle["player_won"] for battle in train], dtype=int)
    catalog_records = train if training_only else train + test
    dex = build_dex(catalog_records)

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
    keys, membership, weights, lookup, species_names = pool_matrix(catalog_records, dex)
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
        "train_ids": [battle["battle_id"] for battle in train],
        "ids": [battle["battle_id"] for battle in test],
        "preparation": {
            "training_only": training_only, "moves": moves, "species": species,
            "slot_columns": slot_columns, "dex_species": sorted(dex), "dex": dex,
            "candidate_teams": [sorted(key) for key in keys],
            "candidate_frequencies": weights.tolist(), "feature_columns": list(A.columns),
        },
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


def _validation_matrices(data, train_idx, valid_idx, *, preparation, groups, te_seed):
    """Rebuild every learned transform at the current fitting boundary."""
    if preparation == "fold":
        if "train_records" not in data:
            raise ValueError("Fold preparation requires raw battles from build_all.")
        train = [data["train_records"][i] for i in train_idx]
        # Do not even pass held-out outcomes to the transformation builder.
        valid = [{k: v for k, v in data["train_records"][i].items() if k != "player_won"}
                 for i in valid_idx]
        local = build_record_features(train, valid, data["chart"], training_only=True)
        fit = np.arange(len(train))
        apply = np.arange(len(train), len(train) + len(valid))
        local_groups = None if groups is None else np.concatenate([groups[train_idx], groups[valid_idx]])
        enc_train, enc_valid = oof_te2(
            local["k1"], local["k2"], local["y"], local["keys"], local["C"],
            fit, apply, seed=te_seed, groups=local_groups,
        )
        A, B = local["A"], local["B"]
        state = local["preparation"]
    else:
        enc_train, enc_valid = oof_te2(
            data["k1"], data["k2"], data["y"], data["keys"], data["C"],
            train_idx, valid_idx, seed=te_seed, groups=groups,
        )
        A, B = data["A"].iloc[train_idx], data["A"].iloc[valid_idx]
        state = data.get("preparation", {"feature_columns": list(A.columns)})
    X_train = pd.concat([A.reset_index(drop=True), enc_train], axis=1)
    X_valid = pd.concat([B.reset_index(drop=True), enc_valid], axis=1)
    return X_train, X_valid, {
        "fitted_state_sha256": digest(state), "n_features": X_train.shape[1],
        "feature_columns": list(X_train.columns),
        "n_moves": len(state.get("moves", [])),
        "n_species": len(state.get("species", [])),
        "n_candidate_teams": len(state.get("candidate_teams", [])),
    }


def cross_validate(data, n_splits=5, *, output_dir=None, reference_oof=None,
                   return_details=False, outer_seed=SEED, model_seed=SEED,
                   te_seed=0, split_mode="random", preparation="global",
                   nested=False, finalists=None, selection_splits=3,
                   selection_seed=17, _indices=None):
    """Evaluate fixed models, optionally selecting fixed blends inside each fold.

    Default return remains a name->accuracy dictionary. With return_details,
    return (report, predictions). Outer/model/encoder seeds are independent.
    Repeated/grouped/fold-local/nested protocols are opt-in; duplicate records
    are always kept together when raw data is available. _indices restricts
    nested selection to the current outer training subset.
    """
    if preparation not in ("global", "fold"):
        raise ValueError("preparation must be global or fold")
    y = np.asarray(data["y"])
    indices = np.arange(len(y)) if _indices is None else np.asarray(_indices)
    ids = pd.Series(data["train_ids"])
    if len(ids) != len(y) or ids.isna().any():
        raise ValueError("train_ids must contain one non-null ID per training row.")
    groups = protocol_groups(data, split_mode)
    if ids.duplicated().any() and groups is None:
        raise ValueError("Duplicate IDs require audited raw battles from build_all.")
    splits = split_indices(y, indices, seed=outer_seed, n_splits=n_splits, groups=groups)
    finalists = tuple(VARIANTS) if finalists is None else tuple(finalists)
    if nested and (not finalists or len(finalists) != len(set(finalists))
                   or any(name not in VARIANTS for name in finalists)):
        raise ValueError("Provide a nonempty unique list of the three fixed submission variants.")
    predictions = {name: np.full(len(y), np.nan) for name in (*MODEL_NAMES, "majority")}
    if nested:
        predictions["nested_selected"] = np.full(len(y), np.nan)
    folds = np.full(len(y), -1, dtype=int)
    seen_p1, seen_p2 = np.zeros(len(y), bool), np.zeros(len(y), bool)
    fold_details, selections = [], []
    has_teams = len(data["k1"]) >= len(y)

    for fold, (train_idx, valid_idx) in enumerate(splits):
        selected = None
        if nested:
            inner_dir = None if output_dir is None else Path(output_dir) / "selection" / f"outer_{fold}"
            inner_report, _ = cross_validate(
                data, n_splits=selection_splits, output_dir=inner_dir,
                return_details=True, outer_seed=selection_seed, model_seed=model_seed,
                te_seed=te_seed, split_mode=split_mode, preparation=preparation,
                _indices=train_idx,
            )
            # Stable, predeclared finalist order breaks ties; no outer score used.
            selected = max(finalists, key=lambda name: inner_report["models"][name]["accuracy"])
            selections.append({
                "outer_fold": fold, "selected": selected,
                "inner_accuracies": {name: inner_report["models"][name]["accuracy"] for name in finalists},
                "inner_assignments_sha256": inner_report["fold_assignments_sha256"],
                "selection_training_row_ids": train_idx.tolist(),
            })
        X_train, X_valid, details = _validation_matrices(
            data, train_idx, valid_idx, preparation=preparation, groups=groups, te_seed=te_seed,
        )
        folds[valid_idx] = fold
        majority_class = int(np.mean(y[train_idx]) > 0.5)
        predictions["majority"][valid_idx] = majority_class
        fold_predictions = {}
        for name in MODEL_NAMES:
            estimator = make_model(name, seed=model_seed)
            estimator.fit(X_train, y[train_idx])
            fold_predictions[name] = estimator.predict_proba(X_valid)[:, 1]
            predictions[name][valid_idx] = fold_predictions[name]
        if nested:
            predictions["nested_selected"][valid_idx] = blend_probabilities(fold_predictions, VARIANTS[selected])
        if has_teams:
            train_p1 = {data["k1"][i] for i in train_idx}
            train_p2 = {data["k2"][i] for i in train_idx if data["k2"][i] is not None}
            seen_p1[valid_idx] = [data["k1"][i] in train_p1 for i in valid_idx]
            seen_p2[valid_idx] = [data["k1"][i] in train_p2 for i in valid_idx]
        details.update({
            "fold": fold, "n_train": len(train_idx), "n_validation": len(valid_idx),
            "train_class_counts": np.bincount(y[train_idx], minlength=2).tolist(),
            "validation_class_counts": np.bincount(y[valid_idx], minlength=2).tolist(),
            "majority_class": majority_class,
            "target_encoding_inner_splits": (5 if groups is None else min(5, len(set(groups[train_idx])))),
            "validation_p1_seen_as_training_p2": int(seen_p2[valid_idx].sum()) if has_teams else None,
        })
        fold_details.append(details)

    for name, weights in VARIANTS.items():
        predictions[name] = blend_probabilities(predictions, weights)
    oof = pd.DataFrame({
        "row_id": indices, "battle_id": ids.iloc[indices].to_numpy(),
        "fold": folds[indices], "player_won": y[indices],
        **{name: values[indices] for name, values in predictions.items()},
    })
    if has_teams:
        oof["p1_seen_in_training"] = seen_p1[indices].astype(int)
        oof["p1_seen_as_p2_in_training"] = seen_p2[indices].astype(int)
    if "audit_metadata" in data:
        meta = data["audit_metadata"]
        oof["duplicate_group"] = meta["duplicate_groups"][indices]
        oof["p1_group"] = meta["team_groups"][indices]
        oof["observation_sha256"] = np.asarray(meta["observation_hashes"])[indices]
    oof = with_decisions(oof)
    report = summarize_validation(oof, reference_oof=reference_oof)
    report.update({
        "schema_version": 2, "outer_fold_seed": outer_seed, "model_seed": model_seed,
        "te_seed": te_seed, "te_requested_splits": 5,
        "te_splitter": "KFold" if groups is None else "StratifiedGroupKFold",
        "n_splits": n_splits, "shuffle": True,
        "threshold": 0.5, "decision_rule": "probability > 0.5; tie -> 0",
        "majority_tie_class": 0, "split_mode": split_mode, "preparation": preparation,
        "splitter": "StratifiedKFold" if groups is None else "StratifiedGroupKFold",
        "duplicate_protection": "audited" if "audit_metadata" in data else "unavailable_without_raw_records",
        "variants": VARIANTS, "model_parameters": json.loads(json.dumps({
            name: make_model(name, seed=model_seed).get_params() for name in MODEL_NAMES
        }), parse_constant=str),
        "fold_details": fold_details, "provenance": provenance(data),
        "fold_assignments_sha256": digest(oof[["row_id", "battle_id", "fold"]].astype(str).values.tolist()),
        "nested_selection": {"enabled": nested, "finalists": list(finalists) if nested else [],
                             "inner_splits": selection_splits, "inner_seed": selection_seed,
                             "tie_rule": "first finalist in declared order", "selections": selections},
    })
    if "audit_metadata" in data:
        duplicates = data["audit_metadata"]["duplicate_groups"]
        if len(set(duplicates[indices])) < len(indices):
            legacy_folds = np.full(len(y), -1, dtype=int)
            for old_fold, (_, old_valid) in enumerate(split_indices(
                    y, indices, seed=outer_seed, n_splits=n_splits)):
                legacy_folds[old_valid] = old_fold
            crossing = sum(len(set(legacy_folds[indices][duplicates[indices] == group])) > 1
                           for group in set(duplicates[indices]))
            report["duplicate_split_effect"] = {
                "components_crossing_original_stratified_folds": crossing,
                "rows_assigned_to_a_different_fold": int((legacy_folds[indices] != folds[indices]).sum()),
                "original_assignments": [{"row_id": int(i), "fold": int(legacy_folds[i])} for i in indices],
                "interpretation": "Original assignments recorded for audit only; duplicate-protected folds used for all reported fits. No rows removed.",
            }
    if has_teams:
        report["composition_subgroups"] = {}
        masks = {"p1_seen_in_training": seen_p1[indices], "p1_absent_as_p1": ~seen_p1[indices],
                 "p1_absent_as_p1_but_seen_as_p2": ~seen_p1[indices] & seen_p2[indices],
                 "p1_absent_on_both_observed_sides": ~seen_p1[indices] & ~seen_p2[indices]}
        for subgroup, mask in masks.items():
            report["composition_subgroups"][subgroup] = {
                "n": int(mask.sum()),
                "models": {name: {"correct": int(oof.loc[mask, f"{name}__correct"].sum()),
                                  "wrong": int((1 - oof.loc[mask, f"{name}__correct"]).sum()),
                                  "accuracy": float(oof.loc[mask, f"{name}__correct"].mean()) if mask.any() else None,
                                  "folds": {str(fold): {
                                      "n": int((mask & (oof["fold"].to_numpy() == fold)).sum()),
                                      "accuracy": (float(oof.loc[mask & (oof["fold"] == fold), f"{name}__correct"].mean())
                                                   if (mask & (oof["fold"] == fold)).any() else None)}
                                      for fold in range(n_splits)}}
                           for name in model_columns(oof)},
            }
    if output_dir is not None:
        write_validation(report, oof, output_dir)
        if "audit" in data:
            _write_json(Path(output_dir) / "data_audit.json", data["audit"])
    if return_details:
        return report, oof
    return {name: metrics["accuracy"] for name, metrics in report["models"].items()}


def _write_json(path, value):
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def repeat_validation(data, output_dir, *, seeds=REPEAT_SEEDS, **options):
    """Describe split sensitivity; never select a seed or pool repeated rows."""
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Repeat seeds must be nonempty and unique.")
    reports = []
    for seed in seeds:
        report, _ = cross_validate(data, output_dir=Path(output_dir) / f"seed_{seed}",
                                   outer_seed=seed, return_details=True, **options)
        reports.append(report)
    summary = {"seeds": list(seeds), "interpretation": "Split sensitivity on the same battles; repetitions are not independent samples; no best seed selected.",
               "models": {}}
    for name in reports[0]["models"]:
        scores = [r["models"][name]["accuracy"] for r in reports]
        deltas = [r["models"][name]["accuracy_delta_vs_reference"] for r in reports]
        summary["models"][name] = {
            "accuracies": scores, "mean_accuracy": float(np.mean(scores)),
            "min_accuracy": min(scores), "max_accuracy": max(scores),
            "descriptive_std": float(np.std(scores)),
            "deltas_vs_reference": deltas, "mean_delta_vs_reference": float(np.mean(deltas)),
        }
    _write_json(Path(output_dir) / "repeated_validation_report.json", summary)
    return summary


def compare_preparation(data, output_dir, **options):
    """Compare information protocols using identical outer groups and models."""
    from pokemon.validation import compare_oof
    global_report, global_oof = cross_validate(
        data, output_dir=Path(output_dir) / "global", preparation="global",
        return_details=True, **options,
    )
    local_report, local_oof = cross_validate(
        data, output_dir=Path(output_dir) / "fold", preparation="fold",
        return_details=True, **options,
    )
    paired = compare_oof(local_oof, global_oof)
    for name in paired:
        paired[name]["accuracy_delta_fold_minus_global"] = (
            local_report["models"][name]["accuracy"] - global_report["models"][name]["accuracy"])
    report = {"models": paired, "reference": "global preparation",
              "interpretation": "Same folds/models; only the information available to fitted transformations differs. No test outcomes used."}
    _write_json(Path(output_dir) / "preparation_comparison.json", report)
    return report


def _final_matrices(data):
    """Create full-train and test matrices with non-leaking target encodings."""
    A, B, y = data["A"], data["B"], data["y"]
    n_train = len(y)

    # --- Training rows use inner out-of-fold values; test rows use all labels.
    enc_train, enc_test = oof_te2(
        data["k1"], data["k2"], y, data["keys"], data["C"],
        np.arange(n_train), np.arange(n_train, n_train + len(B)),
        groups=protocol_groups(data, "random"),
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
