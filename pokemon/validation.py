"""Persist held-out predictions and compare errors on identical battles."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


IDENTITY_COLUMNS = ("battle_id", "fold", "player_won", "row_id", "duplicate_group",
                    "p1_group", "p1_seen_in_training", "p1_seen_as_p2_in_training", "observation_sha256")


def model_columns(oof):
    """Probability columns retain the original model names for compatibility."""
    return [name for name in oof if name not in IDENTITY_COLUMNS
            and not name.endswith(("__prediction", "__correct"))]


def with_decisions(oof):
    """Add explicit decisions/correctness using the unchanged strict threshold."""
    oof = oof.copy()
    for name in model_columns(oof):
        oof[f"{name}__prediction"] = (oof[name] > 0.5).astype(int)
        oof[f"{name}__correct"] = (oof[f"{name}__prediction"] == oof["player_won"]).astype(int)
    return oof


def _checked_oof(oof):
    """Validate a complete OOF table without changing its source row order."""
    if not {"battle_id", "fold", "player_won"}.issubset(oof.columns) or oof.empty:
        raise ValueError("OOF data must contain battle_id, fold and player_won.")
    oof = oof.copy()
    if oof["battle_id"].isna().any():
        raise ValueError("OOF battle IDs must not be null.")
    oof["battle_id"] = oof["battle_id"].astype(str)
    if oof["battle_id"].duplicated().any() and "row_id" not in oof:
        raise ValueError("Duplicate battle IDs require unique source row_id values.")
    if "row_id" in oof and (oof["row_id"].isna().any() or oof["row_id"].duplicated().any()):
        raise ValueError("Source row_id values must be unique and non-null.")
    if not oof["player_won"].isin([0, 1]).all():
        raise ValueError("OOF outcomes must be binary.")
    folds = oof["fold"].to_numpy(dtype=float)
    if (not np.isfinite(folds).all() or (folds < 0).any()
            or (folds != np.floor(folds)).any()):
        raise ValueError("Every battle must have a non-negative integer fold.")
    names = model_columns(oof)
    probabilities = oof[names].to_numpy(dtype=float)
    if (not names or not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError("Every model must have a probability in [0, 1] for every battle.")
    for name in names:
        for suffix, expected in (("__prediction", (oof[name] > 0.5).astype(int)),
                                 ("__correct", ((oof[name] > 0.5) == oof["player_won"]).astype(int))):
            if name + suffix in oof and not np.array_equal(oof[name + suffix], expected):
                raise ValueError(f"Inconsistent saved decisions for {name}.")
    return oof


def _paired_errors(current, previous):
    """Count gross changes as well as the net change in correct predictions."""
    corrected = int((current & ~previous).sum())
    regressed = int((~current & previous).sum())
    return {
        "corrected": corrected,
        "regressed": regressed,
        "net_correct": corrected - regressed,
        "both_correct": int((current & previous).sum()),
        "both_wrong": int((~current & ~previous).sum()),
    }


def compare_oof(current, previous):
    """Compare shared models across runs, aligning by ID rather than row index.

    Reject partial overlap or changed labels/folds instead of silently comparing
    scores on different validation battles. CSV readers should preserve IDs as
    strings, including numeric-looking IDs with leading zeroes.
    """
    current, previous = _checked_oof(current), _checked_oof(previous)
    key = "row_id" if current["battle_id"].duplicated().any() else "battle_id"
    if key not in previous or set(current[key]) != set(previous[key]):
        raise ValueError("OOF runs must contain exactly the same battle IDs.")
    previous = previous.set_index(key, drop=False).loc[current[key]]
    for column in ("battle_id", "fold", "player_won", "observation_sha256"):
        if column not in current or column not in previous:
            continue
        if not np.array_equal(current[column].to_numpy(), previous[column].to_numpy()):
            raise ValueError(f"OOF runs have different {column} values for the same battles.")
    shared = [name for name in model_columns(current) if name in model_columns(previous)]
    if not shared:
        raise ValueError("OOF runs have no model probability columns in common.")
    y = current["player_won"].to_numpy()
    return {
        name: _paired_errors(
            (current[name].to_numpy() > 0.5) == y,
            (previous[name].to_numpy() > 0.5) == y,
        )
        for name in shared
    }


def summarize_validation(oof, reference="sub1_lgbm", reference_oof=None):
    """Summarize all models and compare them with a named same-run reference."""
    oof = _checked_oof(oof)
    if reference not in oof or reference in IDENTITY_COLUMNS:
        raise ValueError(f"Unknown reference model: {reference!r}.")
    y = oof["player_won"].to_numpy()
    folds = oof["fold"].to_numpy()
    reference_correct = (oof[reference].to_numpy() > 0.5) == y
    models = {}
    for name in model_columns(oof):
        correct = (oof[name].to_numpy() > 0.5) == y
        models[name] = {
            "accuracy": float(correct.mean()),
            "correct": int(correct.sum()),
            "wrong": int((~correct).sum()),
            "accuracy_delta_vs_reference": float(correct.mean() - reference_correct.mean()),
            "fold_counts": {
                str(int(fold)): {"correct": int(correct[folds == fold].sum()),
                                 "wrong": int((~correct[folds == fold]).sum()),
                                 "vs_reference": _paired_errors(correct[folds == fold], reference_correct[folds == fold]),
                                 "accuracy_delta_vs_reference": float(correct[folds == fold].mean() - reference_correct[folds == fold].mean())}
                for fold in sorted(set(folds))
            },
            "fold_accuracies": {
                str(int(fold)): float(correct[folds == fold].mean())
                for fold in sorted(set(folds))
            },
            "vs_reference": _paired_errors(correct, reference_correct),
        }
    report = {
        "n_observations": len(oof),
        "reference_model": reference,
        "fold_sizes": {str(int(fold)): int((folds == fold).sum())
                       for fold in sorted(set(folds))},
        "models": models,
    }
    if reference_oof is not None:
        previous = pd.read_csv(
            reference_oof, dtype={"battle_id": str}, keep_default_na=False,
            float_precision="round_trip",
        )
        report["previous_run"] = {
            "oof_file": str(reference_oof),
            "models": compare_oof(oof, previous),
        }
    return report


def write_validation(report, oof, output_dir):
    """Save a JSON report and one row per battle with all class-1 probabilities."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    oof_path = output_dir / "validation_predictions.csv"
    report["oof_file"] = oof_path.name
    with_decisions(oof).to_csv(oof_path, index=False)
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
