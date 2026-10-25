# Interpreting the supplied Kaggle execution log

The run completed successfully: all three submission files were generated,
each contained 5,000 predictions, and template-order and binary-label checks
passed. The debugger's frozen-module messages and the `mistune`/`nbconvert`
escape-sequence warnings did not abort execution.

## Measured results

| Model / variant | Historical reference | Supplied Kaggle log |
| --- | ---: | ---: |
| LightGBM | 69.50% | 69.50% |
| XGBoost | 69.47% | 69.29% |
| CatBoost | 69.31% | 69.31% |
| LightGBM / XGBoost, 50/50 | 69.70% | 69.46% |
| LightGBM / XGBoost / CatBoost, 45/45/10 | 69.79% | 69.50% |

The historical ensemble result was not reproduced in the supplied run.
An ensemble can lose accuracy when a constituent model's predictions change;
its performance is measured from the averaged probabilities, not by averaging
the individual accuracies.

`requirements.txt` specifies minimum versions, so different installations can
use different releases. Platform differences can also affect numerical tree
training. Neither possibility is confirmed as the specific cause here: the
supplied log does not identify the installed library versions, repository
commit or input checksums. The updated notebook records those details instead
of promising an exact historical score.

The three clone messages do not by themselves establish that the clone command
ran three times; the successful output contains no clone failure. Setup now
reuses an existing checkout and prints its commit. Reusing a checkout does
not update it: use a fresh Kaggle session or explicitly update the repository
when testing new source code.

## Updated notebook outputs

The notebook saves `run_manifest.json` with:

- Loaded and installed versions of NumPy, pandas, scikit-learn, LightGBM,
  XGBoost and CatBoost, plus Python and platform details.
- Repository commit, SHA-256 of each official input, feature column order and
  a feature-schema checksum.
- Model parameters, validation settings and measured scores.

It checks for a mismatch between loaded and installed library versions, which
requires restarting the kernel after a dependency installation. Submission
explanations now refer to scores measured in the current run.

The original run's **637 columns** identify the baseline. The newer
active-strength candidate adds **42 columns**, giving **679 before team target
encodings**, and was not executed in the supplied log. The notebook now includes
a separate research section, enabled by `RUN_STRENGTH_RESEARCH = True`, which:

1. Evaluates the seed-42 XGBoost strength candidate and the original reference
   on the same five folds with nested target encodings.
2. Saves `strength_results.json` and `strength_oof.csv.gz`.
3. Produces and validates `strength_candidates/candidate_strength_xgb.csv`
   and `strength_candidates/candidate_strength_blend.csv`.

The local candidate score of 69.85% is a previous measurement, not a promised
Kaggle result. Creating candidate files does not select additional competition
submissions, and none of these changes establishes accuracy above 70%.
