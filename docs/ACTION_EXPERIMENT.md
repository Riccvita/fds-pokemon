# Predicting Player Actions from Observed Battle Prefixes

## Research question

Can an auxiliary classical machine-learning model learn each player's next
battle action from the first thirty observed turns, and can its probabilities
improve final-battle winner predictions?

The existing solution achieves **69.79% five-fold out-of-fold accuracy**.
The purpose of this experiment is to learn a different signal, rather than
continue tuning the existing gradient-boosting hyperparameters.

## Supervision and features

The action learner predicts three mutually exclusive classes:

- `damage`: the recorded move belongs to the physical or special category.
- `status`: the recorded move belongs to the status category.
- `unrecorded`: no move is recorded. This class may include switches, forced
  replacements, or other missing actions; it is **not** a reliable switch label.

Only official `train.jsonl`, `test.jsonl`, and `type_chart_gen2.csv` inputs are
allowed. No external game logs, strategy tables, pretrained models, or neural
networks are used. The winner label is not used for the auxiliary action model.

A causal snapshot at cutoff `t` describes the observed turns up to and including
`t`. Its target is the recorded action on turn `t+1`. The training cutoffs are
**9, 14, 19, 24, and 29**, yielding **10 labeled actions per battle** (two
players at five cutoffs). The final inference snapshot uses the complete
observed turn-30 prefix; no post-turn-30 action is known or used.

The 69 input attributes include:

1. Remaining visible team health, fainted counts, active health and status.
2. Active species base stats, stages, and current matchup characteristics.
3. Historical damage, status, and unrecorded-action rates over multiple windows.
4. Revealed moves of each active species and their approximate matchup strength.
5. Recent species replacements and observed action choices.

The action classifier is a regularized **LightGBM multiclass** model. Its
predicted class probabilities are then transformed into player-specific
probabilities, P1–P2 differences, action-clash probabilities, and expected
attack-pressure proxies. These sixteen derived features are appended to the
original winner-prediction matrices.

## Leakage-safe validation

The experiment uses the same five outer `StratifiedKFold` splits as the
original solution (seed 42, 10,000 training battles). For each outer fold:

1. Train the action classifier only on earlier-turn action observations from
   **outer-training battles**. Never train it on action labels from outer
   validation battles.
2. Generate action-probability features for outer training and validation
   using the action classifier from step 1.
3. Recompute winner-label target encoding using **inner out-of-fold**
   statistics on outer training, and outer-train-only statistics on validation.
4. Train the winner classifier and score only the outer-validation battles.

Results reported below use all five folds, each with 2,000 held-out battles.
The competition test set and public leaderboard did not influence the model
selection or reported results.

## Results

| Model / combination | Five-fold accuracy | Five-fold ROC AUC |
|---|---:|---:|
| Original LightGBM + XGBoost + CatBoost (45/45/10) | **69.79%** | 0.76559 |
| Original LightGBM with action features | 69.28% | 0.76532 |
| Original XGBoost with action features (220 trees) | 69.38% | 0.76387 |
| Original ensemble + 15% action-LightGBM | **69.80%** | 0.76598 |
| Original ensemble + 30% action-LightGBM | 69.78% | 0.76624 |
| Original ensemble + 50% action-LightGBM | 69.77% | 0.76633 |
| Original ensemble + 15% action-XGBoost | 69.68% | 0.76583 |
| Original ensemble + 30% action-XGBoost | 69.64% | 0.76591 |
| Original ensemble + 50% action-XGBoost | 69.66% | 0.76572 |

The action classifier itself achieves **62.34%** mean out-of-fold multiclass
action accuracy, compared with a **50.29%** fold-wise majority-class
baseline. Thus the behavior prediction task is learnable from the observed
prefix, but this information is largely redundant for winner prediction.

The +15% action-LightGBM blend yields **6,980 correct battles vs. 6,979** for
the existing ensemble. Compared with the original classifier, it fixes
**39 mistakes** but introduces **38 new errors**. The change is only **one
battle out of 10,000** and is not meaningful evidence of better generalization.
The marginal ROC AUC gains do not override the competition's accuracy metric.

**Decision:** Retain the existing three competition submissions. The action
experiment is documented research, not a replacement model. The goal of 80%
accuracy was not reached.

## Reproduction

Install dependencies from `requirements.txt`, place the official competition
files in a local data directory, and run:

```bash
python action_experiment.py \
  --data /path/to/competition_data \
  --include-xgb \
  --output docs/ACTION_CV_RESULTS.json
```

This rebuilds the original baseline features and original OOF predictions, then
runs five-fold action-model validation. Depending on the environment, complete
reproduction can be computationally expensive. To reuse previously calculated
features and original OOF predictions, pass:

```bash
python action_experiment.py \
  --data /path/to/competition_data \
  --baseline-cache /path/to/baseline.joblib \
  --reference-oof /path/to/original_oof.joblib \
  --action-cache /path/to/action_inputs.joblib \
  --include-xgb
```

The original reference predictions must be out-of-fold outputs of the models
in `pokemon/main.py`, using the same dataset order and outer splits. The script
supports resumable checkpoints for completed folds. Run unit tests with:

```bash
python -m unittest discover -s tests -v
```

The original Kaggle notebook and three selected submissions remain unchanged.
This action experiment only uses the training data for evaluation and is not
silently substituted into any final submission.

## Limitations and potential follow-ups

The label `unrecorded` combines heterogeneous game decisions. Some instances
represent switches, but the state and move logs do not always distinguish
voluntary switches from forced replacements and other missing actions. The
simple offense-pressure proxy is not a damage simulator and does not model
move choice conditionally. More faithful action annotations, if recoverable
from the *provided* battle records alone, could enable a more informative
behavior model. Any follow-up should be compared with the original 69.79%
model under leakage-safe held-out validation, not tuned on competition test
outcomes or frequent public-leaderboard feedback.
