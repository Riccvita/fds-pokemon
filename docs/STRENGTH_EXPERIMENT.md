# Active-strength and seed-averaging experiments

## Outcome

**70% accuracy was not reached.** The strongest new individual model is
XGBoost with 42 deterministic active-strength interactions: **69.85%**
five-fold out-of-fold accuracy (6,985 correct predictions out of 10,000).
Five-seed averaging of that model scores **69.77%**.

The original three-model ensemble scores **69.42% in this environment**.
The historical **69.79%** in the earlier reports was not reproduced here.
Consequently, the within-run gain for the new individual XGBoost is 0.43
percentage points; comparing 69.85% directly with the historical result would
mix runs. The exact package versions and source-data checksum are saved with
the new results. These experiments do not replace the three selected
submissions or edit the Kaggle notebook.

## Representation

`pokemon/strength_features.py` combines statistics already present in the
original feature matrix:

- Approximate active attack, defense, special attack, special defense and
  speed after stat-stage modifiers, burn and paralysis.
- Offense and defensive bulk weighted by remaining active HP.
- Physical and special attack-to-defense ratios and health-normalized pressure.
- Physical and special endurance and P1-minus-P2 differences.

The 42 interactions help shallow trees represent multiplicative effects.
They are approximate indicators, not simulated damage or exact game stats.
No winner label, battle ID, unobserved future turn, external database or
additional replay is used to compute them. Missing species stats remain NaN.

## Validation and interpretation

Every experiment uses all 10,000 original training battles, the same
`StratifiedKFold(5, shuffle=True, random_state=42)` split, and the original
nested team target encoder. Outer validation labels never enter that fold's
training features or model fits. The threshold remains 0.5.

The five seeds, averaged with equal weights, are 42, 137, 2026, 31415 and
271828. Seeds were not selected individually by validation performance.
The default ensemble coefficients remain 45% LightGBM, 45% XGBoost and
10% CatBoost.

| Experiment | OOF accuracy |
| --- | ---: |
| Original three-model ensemble, current run | 69.42% |
| Original ensemble averaged over five seeds | 69.66% |
| Original features plus HP-precision indicator | 69.53% |
| Separate models for rounded/high-precision HP logs | 68.38% |
| CatBoost with native species/roster/move categories | 69.55% |
| Wider LightGBM, 700 trees / 12 leaves | 69.29% |
| Histogram gradient boosting | 69.08% |
| ExtraTrees | 68.06% |
| Last-appearance/retired-Pokemon features, original blend | 69.22% |
| Active-strength features, original blend | 69.63% |
| **Active-strength XGBoost, seed 42** | **69.85%** |
| Active-strength XGBoost, five-seed average | 69.77% |
| Active-strength XGBoost, 800 trees / learning rate 0.02 | 69.69% |
| Active-strength XGBoost, 800 trees / depth 3 | 69.64% |
| Active-strength XGBoost, 600 trees / depth 5 / stronger regularization | 69.54% |
| Active-strength XGBoost, reduced column sampling / stronger regularization | 69.53% |

Additional exploratory blends used weights 25%, 50% and 75% for the new
model against either the original ensemble or its five-seed average. The
highest such selection score was **69.89%**, combining 75% seed-42
active-strength XGBoost and 25% of the original five-seed ensemble. This
coefficient was evaluated on the same OOF predictions used to report the
score; it is not an independent estimate and was not promoted to a selected
submission.

Both feature families and model profiles have been compared on these same
folds. All quoted maxima are exploratory model-selection results, not
untouched test performance. A few additional correct battles around 70%
would not establish a reliable generalization gain. Test outcomes and
leaderboard feedback were not used.

## Reproduction

The focused strength experiment is independently executable:

```bash
python strength_experiment.py --data /path/to/competition

# Reproduce the strongest individual candidate and optionally write its CSVs.
python strength_experiment.py --data /path/to/competition --seeds 42 \
    --out docs/STRENGTH_SINGLE_CV_RESULTS.json --submission

# Repeat one of the explicitly defined parameter comparisons.
python strength_experiment.py --data /path/to/competition --seeds 42 \
    --profile shallow --out docs/STRENGTH_SHALLOW_CV_RESULTS.json

POKEMON_DATA_DIR=/path/to/competition python -m unittest discover -s tests -v
```

The script saves per-fold accuracy, AUC, corrected/regressed counts against
the within-run reference, package versions, the training file's SHA-256,
and aligned OOF probabilities. Candidate submissions are written separately
under `submissions/candidates/`; they preserve the original test ID order.
The reference blend in the focused script is always the original seed-42
ensemble. The wider screening table above also includes separate experiments
with other feature representations and estimators; those are not all exposed
by this focused command.

Unit tests cover row-order preservation, label independence, stat stages,
status interactions, zero-HP denominators, missing stats and input immutability.
