# Research Note: Attempts to Reach 70% Accuracy

## Research objective

Improve the benchmark for the Sapienza Pokémon Battle Prediction challenge from
**69.79%** five-fold out-of-fold accuracy to at least **70.00%** without
external data, neural networks, or leakage.

**Outcome:** The threshold was not reached. The original three-model ensemble
remains the most reliable submission candidate. It would be scientifically
incorrect to claim a 70% score based on the experiments below.

## Validation protocol

All five-fold experiments use the same `StratifiedKFold(5, shuffle=True,
random_state=42)` division of the 10,000 labeled battles. The existing
team-win target encoding is cross-fitted inside each outer training split.
Models never see held-out labels while fitting. Their predictions are placed
back at original training row positions and evaluated once across all folds.

The Kaggle test set is **not scored or used for model selection**. Unlabeled
test team identities are used in the original baseline's candidate-pool
feature builder, consistent with earlier pipeline behavior.

## Five-fold results

| Experiment | OOF accuracy | AUC, when computed |
| --- | ---: | ---: |
| Original LightGBM | 69.50% | — |
| Original XGBoost | 69.47% | — |
| Original CatBoost | 69.31% | — |
| Original three-model ensemble (45/45/10) | **69.79%** | 0.76559 |
| Shallower XGBoost (750 estimators, depth 3) | 69.29% | 0.76566 |
| Original + 30% shallower XGBoost | 69.57% | 0.76619 |
| XGBoost + roster-pair target encodings | 69.23% | 0.76452 |
| Original + 15% pair-encoded XGBoost | 69.71% | 0.76625 |
| LightGBM with top 70 features selected inside each fold | 69.07% | 0.76278 |
| XGBoost with top 120 features selected inside each fold | 69.45% | 0.76521 |
| Original + 10% selected-feature XGBoost | 69.74% | 0.76577 |
| Group-dependent threshold, cross-fitted (exploratory) | 69.81% | — |

The group-dependent threshold changes only **two predictions per 10,000**
relative to the original reference. Several related decision rules were tried;
this is not strong evidence of out-of-sample improvement. We have therefore
not adopted the group threshold as a final submission.

**Additional first-fold-only screen:** One-hot encoding of the initial and
final active Pokémon (`identity_features.py`) achieved 68.95% with LightGBM,
68.65% with XGBoost and 68.35% with CatBoost, against the 69.20% original
ensemble on that same fold. We did not claim a five-fold score for this feature
family.

## Details of the new supervised features

`pokemon/pair_te.py` estimates three kinds of win-rate priors from labeled
training battles:

1. **Species-level prior:** historical outcome of a given Pokémon on the
   observed side, regularized toward 50%.
2. **Teammate pair prior:** historical outcome of each unordered pair of
   Pokémon observed on the same team.
3. **Cross-team pair prior:** historical result for an ordered attacker-side
   Pokémon and opposing Pokémon pair.

For Player 1, all six species are known. For Player 2, only the species that
have appeared within the supplied 30 turns are included. The target is
reversed when recording the opponent's win-rate statistics. Inner folds
prevent the true winner of an observation from influencing the features of
that same observation.

Although these priors contain real strategic information, the resulting
XGBoost did not exceed the original model's accuracy. A modest AUC increase
when blended did not translate into additional correct hard predictions.

## Other approaches evaluated

- **Threshold calibration:** The fixed 0.50 cutoff was competitive across
  the original ensemble's held-out probabilities. Re-optimizing blend weights
  and thresholds on other folds did not give a stable lift.
- **Feature selection:** Fold-local LightGBM feature importance was used to
  filter noisy inputs before fitting a new estimator. The best tested filtered
  variant still underperformed the original ensemble.
- **Alternative tree depth:** Shallower XGBoost had comparable ROC AUC but
  worse classification accuracy.
- **Species identity:** Lead and end-of-observation categorical indicators
  did not improve the initial validation fold.
- **Kernel SVM:** A preliminary RBF-SVM model on the 75 highest-ranked
  features was below the baseline and did not justify full CV.

Previously completed action modeling and transition-modeling experiments are
recorded in `ACTION_EXPERIMENT.md` and `TRANSITION_EXPERIMENT.md`.

## Reproduction

The codebase includes English docstrings and comments, plus the executable
research command:

```bash
python research_70.py --data /path/to/competition --method pair-xgb \
    --out docs/pair_cv_results.json

python research_70.py --data /path/to/competition --method selected-xgb \
    --out docs/selected_cv_results.json

python -m unittest discover -s tests -v
```

The first two commands reconstruct all numerical features from the provided
input files, re-create the nested encodings, and run five-fold validation.
They are not used for generating the three official submissions.

## Recommendation

Keep the original validated submission variants and describe these experiments
transparently in the methodology report. An observed accuracy of 69.79% does
**not** imply that the private leaderboard will show the same score, nor does
it establish an upper bound of 70% on the underlying task.
