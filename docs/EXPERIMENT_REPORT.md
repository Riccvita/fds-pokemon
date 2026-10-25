# Experiment Report: Pokémon Battle Prediction

## 1. Research objective and constraints

Predict the final winner (`player_won`) of Gen 2 OU Pokémon battles from Player 1's six Pokémon, Player 2's lead, and the first 30 active turns. Training consists of 10,000 labeled battles; the competition test data contains 5,000 unlabeled battles.

**Original target:** 80% classification accuracy. **Best observed five-fold out-of-fold accuracy:** 69.79%. The target has **not** been met. No conclusions about the private leaderboard can be drawn from training-set cross-validation alone.

All tested final models are classical gradient-boosting methods. No neural networks, external species databases, extra battle logs, pretrained models or additional labels were used.

## 2. Validation methodology

The main comparison uses five outer folds from `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Each outer fold trains on 8,000 battles and validates on 2,000. Predictive performance is computed from concatenated out-of-fold (OOF) predictions, covering exactly 10,000 unique training examples.

Team win-rate features are computed using **nested** target encoding:

1. On each outer training subset, fit the target encoder using inner K-fold cross-fitting. Every training row is encoded without using its own label.
2. Fit the outer-fold model using these inner OOF features.
3. Encode the outer validation subset using labels from the corresponding outer training subset **only**.
4. Produce predictions for the held-out outer validation subset and assemble the full OOF vector.

The unlabeled competition test set is not used for tuning or scoring. The original baseline does use unlabeled team identities from both provided train and test records to define its candidate-team pool, a transductive feature-building choice. No test labels are used.

## 3. Feature engineering

The validated reference pipeline creates 637 numerical, non-target-encoded features. These include:

- Team stats and type effectiveness against the opposing lead.
- Held item flags and observed Pokémon/move occurrences.
- HP fractions, status conditions, temporary boosts and active-battle effects.
- Total remaining team HP, estimated KO counts, healing events and surviving roster strength.
- Observed damage, switch counts, and active-HP advantage trends.
- Candidate-team inference of P2's unseen Pokémon using supplied team identities.
- Extra leakage-safe supervised team-rating encodings added inside each fold.

### Additional experimental features

A separate `pokemon/features5.py` module investigates aggregated party-HP timelines, changes in effective health and statuses, move attack pressure, and individual species health indicators. These were tested on the first outer fold, rather than across all five folds. Adding them did **not** improve validation accuracy. Keeping the final pipeline unchanged reduces complexity and ensures that results are reproducible.

## 4. Final model comparisons

| Model | Training OOF accuracy | Notes |
| --- | ---: | --- |
| LightGBM | 69.50% | 350 estimators, 8 leaves, strong regularization |
| XGBoost | 69.47% | 500 estimators, depth 4, histogram-based trees |
| CatBoost | 69.31% | 500 iterations, depth 5 |
| LightGBM 50% + XGBoost 50% | 69.70% | Probability average |
| LightGBM 45% + XGBoost 45% + CatBoost 10% | **69.79%** | Highest tested OOF accuracy |

The three-model blend improves 29 of 10,000 outcomes relative to LightGBM alone: 123 previously incorrect predictions become correct, while 94 previously correct predictions become incorrect.

### Outer-fold accuracy comparison

| Fold | LightGBM | Final three-model blend |
| --- | ---: | ---: |
| 1 | 69.10% | 69.20% |
| 2 | 68.65% | 69.15% |
| 3 | 69.40% | 70.30% |
| 4 | 71.00% | 70.95% |
| 5 | 69.35% | 69.35% |
| **OOF aggregate** | **69.50%** | **69.79%** |

These differences are small. The blend coefficients were compared on the same OOF set used for reporting, so the 69.79% figure is a **selection result**, not a genuinely untouched independent test score. The approximate binomial sampling uncertainty of a 69.79% proportion at 10,000 samples is around ±0.9 percentage points (95%); this simplified interval does not account for model-selection uncertainty or correlated errors.

## 5. Experiments without a robust gain

### Expanded temporal representation

Several combinations of approximately 450 more temporal/species features were investigated on outer fold 1:

| Configuration | Fold 1 accuracy |
| --- | ---: |
| Original LightGBM | 69.10% |
| Temporal features excluding species-specific indicators | 68.95% |
| All temporal features, original-style parameters | 68.50% |
| All temporal features, deeper configuration | 68.95% |

No additional cross-validation was spent on a consistently weaker first-fold feature set.

### Opposing-team candidate coverage

The original candidate-team lookup is constructed from the visible six-member P1 teams. This omits some otherwise known P2 team configurations. Enlarging the pool with fully revealed P2 lineups reduced the number of cases without candidates (from 3,272 to 1,107 across 15,000 train-plus-test battles) but slightly worsened fold-1 accuracy (69.10% to 68.80%). It was **not** adopted without further evidence.

### Regularized species ratings

A logistic model based on the difference between the players' estimated team-species vectors achieved about 56% accuracy on fold 1, substantially below the gradient-boosting models.

### Earlier baseline investigations

Earlier experiments considered alternative LightGBM hyperparameters, approximate knockouts and direct per-turn time-series features. None yielded a material cross-validated gain. This fact is reported descriptively because the detailed earlier experiment logs are not part of the new five-fold run.

## 6. Interpretation of the accuracy ceiling

The provided 30 turns only offer a partial view of battles that can last up to 80 turns. Predicting the eventual result remains uncertain, especially when both teams retain several viable members. This is a plausible contributor to the observed accuracy plateau, **not** a proof that 80% is mathematically impossible.

Reaching 80% would require new, clearly measured predictive information or a materially stronger representation, not merely replacing one gradient-boosting implementation with another.

## 7. Submission and reproducibility

The three reproducible submissions are:

1. `sub1_lgbm.csv` — LightGBM only.
2. `sub2_lgb_xgb.csv` — equally weighted LightGBM and XGBoost.
3. `sub3_lgb_xgb_cat.csv` — LightGBM/XGBoost/CatBoost at 45/45/10.

Each file has 5,000 rows, matching the provided sample's column names and `battle_id` order, with predictions strictly equal to 0 or 1. They were regenerated on the full labeled training data using an inner-OOF training target encoder. Output correctness was verified programmatically. The original LightGBM submission reproduced exactly.

```bash
python -m pip install -r requirements.txt
python run.py --data /path/to/competition_files --out submissions --cv
```

The public Kaggle notebook must clone the GitHub repository, regenerate the three chosen submissions, and include a brief explanation before each one. Competition end: 5 November 2026, 23:59. Notebook and written report deadline: 12 November 2026.
