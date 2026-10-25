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
first supplied log did not identify the installed library versions, repository
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

## Follow-up Kaggle run with the strength section enabled

The second supplied log identifies commit
`93ff2eaf4d95b50a7cf3ec47d26e2908b306bc38` and successfully executes the
679-column candidate. It reports:

| Candidate | Five-fold OOF accuracy | Correct out of 10,000 |
| --- | ---: | ---: |
| Original three-model reference | 69.50% | 6,950 |
| Strength XGBoost, seed 42 | **69.64%** | **6,964** |
| Fixed 50/50 strength/reference blend | 69.51% | 6,951 |

The strength-fold accuracies are 69.15%, 68.55%, 70.15%, 71.20% and 69.15%.
Their aggregate is 69.64%; folds above 70% do not establish aggregate accuracy
above 70%. The improvement over the reference is 0.14 percentage points,
or 14 net additional correct battles. It is a small validation gain, not
evidence of a reliable leaderboard improvement.

The direct package versions are NumPy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1,
LightGBM 4.6.0, CatBoost 1.2.10 and XGBoost 3.4.1. XGBoost and CatBoost have
the same versions as the local 69.85% run; the other four differ. The log
therefore does not support attributing the discrepancy specifically to an
XGBoost-version change. Its input checksums and OOF file have not been supplied.

The observed scores and version information are recorded, with their source
identified, in `KAGGLE_STRENGTH_REPORTED_RESULTS.json`. The six direct versions
are available in `requirements-kaggle.txt` for controlled comparisons:

```bash
python -m pip install -r requirements-kaggle.txt
python strength_experiment.py --data /path/to/competition --seeds 42 \
    --out /tmp/strength_kaggle_versions.json
```

This pins direct packages, not transitive dependencies or operating-system
libraries, and does not guarantee an identical score on another platform.

## Controlled local comparison

A separate local execution was performed with **all six reported Kaggle direct
package versions**. It still measured **69.85%** for strength XGBoost and
69.42% for the reference. The regenerated train/test feature values, column
order and labels exactly matched the earlier local run. All hard OOF
predictions matched; probability differences were at most 2.23e-16.
The recorded result is `STRENGTH_MATCHED_VERSIONS_LOCAL_RESULTS.json`.

Thus changing those six direct versions did not reproduce or explain the
Kaggle discrepancy in this controlled local comparison. Local execution is
on macOS/ARM; the supplied Kaggle log does not include its manifest's platform
and input checksums. Remaining possibilities include platform/build or
transitive-library differences and unverified input/schema differences.
The exact cause is not established. There is no evidence here that simply
pinning these versions will restore 69.85% on Kaggle.
