"""Reproduce classical modeling experiments targeting 70% battle accuracy.

No predictions from Kaggle test battles or private leaderboard scores are used
when computing validation metrics. The original fold boundaries are preserved.

Examples
--------
python research_70.py --data /path/to/competition --method pair-xgb
python research_70.py --data /path/to/competition --method selected-xgb
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from pokemon.features import read_jsonl
from pokemon.identity_features import build_identity_features
from pokemon.main import SEED, build_all, make_model
from pokemon.pair_te import extract_rosters, oof_pair_features
from pokemon.te import oof_te2


METHODS = ('pair-xgb', 'identity-lgb', 'identity-xgb',
           'selected-lgb', 'selected-xgb', 'shallow-xgb')


def fit_experimental_model(name, features, labels, validation_features):
    """Train a candidate model without accessing the validation targets.

    Parameters
    ----------
    name : str
        Experiment selected from METHODS.
    features : pandas.DataFrame
        Outer-training model matrix.
    labels : numpy.ndarray
        Labels of the outer-training rows only.
    validation_features : pandas.DataFrame
        Features of held-out rows, with identical columns.

    Returns
    -------
    numpy.ndarray
        Held-out class-one probabilities.
    """
    if name == 'shallow-xgb':
        estimator = XGBClassifier(
            n_estimators=750,
            max_depth=3,
            learning_rate=0.03,
            colsample_bytree=0.75,
            subsample=0.80,
            min_child_weight=8,
            reg_lambda=12,
            tree_method='hist',
            n_jobs=2,
            random_state=SEED,
        )
    elif name.endswith('-lgb'):
        estimator = make_model('lgb')
    else:
        estimator = make_model('xgb')

    # --- Feature selection is fitted within the current outer-training fold.
    if name.startswith('selected-'):
        selector = make_model('lgb')
        selector.fit(features, labels)

        n_columns = 70 if name == 'selected-lgb' else 120
        ranked_columns = np.argsort(selector.feature_importances_)[::-1]
        selected_columns = features.columns[ranked_columns[:n_columns]]

        features = features[selected_columns]
        validation_features = validation_features[selected_columns]

    estimator.fit(features, labels)
    return estimator.predict_proba(validation_features)[:, 1]


def validate_experiment(data, method, battles):
    """Compute five-fold accuracy with nested team and pair target encodings."""
    if method not in METHODS:
        raise ValueError(f'Unknown research method: {method}')

    # --- Precompute label-independent information from the supplied battles.
    y = data['y']
    out_of_fold = np.zeros(len(y), dtype=float)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rosters = extract_rosters(battles) if method == 'pair-xgb' else None

    if method.startswith('identity-'):
        identity_features, _ = build_identity_features(battles)
    else:
        identity_features = None

    fold_scores = []

    for fold, (train_idx, validation_idx) in enumerate(folds.split(data['A'], y), start=1):
        # --- Existing supervised team encodings are cross-fitted within the fold.
        train_te, validation_te = oof_te2(
            data['k1'], data['k2'], y, data['keys'], data['C'],
            train_idx, validation_idx,
        )

        X_train = pd.concat([
            data['A'].iloc[train_idx].reset_index(drop=True), train_te,
        ], axis=1)
        X_valid = pd.concat([
            data['A'].iloc[validation_idx].reset_index(drop=True), validation_te,
        ], axis=1)

        # --- Pair statistics are also computed only from the outer training rows.
        if rosters is not None:
            pair_train, pair_valid = oof_pair_features(
                rosters, y, train_idx, validation_idx,
            )
            X_train = pd.concat([X_train, pair_train], axis=1)
            X_valid = pd.concat([X_valid, pair_valid], axis=1)

        # --- Species identities are unsupervised and therefore label-independent.
        if identity_features is not None:
            X_train = pd.concat([
                X_train, identity_features.iloc[train_idx].reset_index(drop=True),
            ], axis=1)
            X_valid = pd.concat([
                X_valid, identity_features.iloc[validation_idx].reset_index(drop=True),
            ], axis=1)

        probabilities = fit_experimental_model(method, X_train, y[train_idx], X_valid)
        out_of_fold[validation_idx] = probabilities

        score = accuracy_score(y[validation_idx], probabilities > 0.5)
        fold_scores.append(float(score))
        print(f'Fold {fold}: {score:.4%}', flush=True)

    return {
        'method': method,
        'n_observations': int(len(y)),
        'fold_accuracies': fold_scores,
        'accuracy': float(accuracy_score(y, out_of_fold > 0.5)),
        'roc_auc': float(roc_auc_score(y, out_of_fold)),
        'test_labels_used': False,
    }


def main():
    """Read official inputs, validate a selected method, and write its score."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--out', type=Path, default=Path('docs/research_70_results.json'))
    arguments = parser.parse_args()

    data = build_all(arguments.data)
    battles = read_jsonl(arguments.data / 'train.jsonl')
    results = validate_experiment(data, arguments.method, battles)

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(f"Final validation accuracy: {results['accuracy']:.4%}", flush=True)


if __name__ == '__main__':
    main()
