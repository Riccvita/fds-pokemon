# Learning Battle-State Transitions: Research Experiment

**Task:** Predict the final outcome of a Pokémon Gen 2 OU battle from its first 30 observed turns.  
**Data:** Only the official 10,000 labeled and 5,000 unlabeled competition battles and supplied Generation 2 type chart.  
**Metric:** Binary classification accuracy.  
**Validation:** Five outer stratified folds, seed 42; supervised target encoding is nested within each outer training fold.

## Research question

Could a model that *learns how battles evolve*, rather than merely summarizing the first 30 turns, improve the original 69.79% classification ensemble?

We built two **classical machine-learning auxiliary tasks** from states that are already visible in the supplied data. Neither auxiliary learner is trained to predict the winner. No external data, pretrained models, neural networks, or hidden turns are used.

## Experiment 1: One-turn health transitions

The competition records expose the state of the active Pokémon after each observed turn, together with the move used. We create a training example when a defender remains the same species in consecutive recorded states and the opposing player used a damaging move on the current turn.

The auxiliary target is the **observed net HP reduction**:

\[
\Delta h = h_{t-1}^{\text{defender}}-h_t^{\text{defender}}.
\]

This is a noisy outcome proxy: passive recovery, residual status damage, failed attacks, and other effects can influence it. It is *not* a perfect physical damage simulator.

### Features

- Attacking move's power, accuracy, priority, and physical/special category.
- Type multiplier from the official Generation 2 type chart, STAB approximation, and observed Pokémon base stats.
- Attacker/defender relative stats, active HP and status conditions, stat boosts, speed, and turn number.

A regularized LightGBM regressor is trained exclusively on state changes from each outer training fold. On the first held-out outer fold, it obtained approximately **R² = 0.415** for net-health-change prediction. This result describes the auxiliary regression task, **not winner-prediction accuracy**.

### Estimated endgame damage pressure

At turn 30, we identify remaining Pokémon, their last observed HP and attacking moves. We predict the possible damage of each *observed move* against each *revealed* remaining opponent. Aggregates include maximum/average potential damage, potential knockout fraction, damage relative to remaining HP, active-matchup pressure, and coverage counts. Pokémon whose moves have not been observed are not assigned invented moves. Unrevealed opponents are not treated as fainted.

## Experiment 2: Ten-turn resource forecasting

For each battle, we create prefix snapshots at turns **5, 10, 15, and 20**, then observe how estimated team health changes ten turns later (turns 15, 20, 25, and 30, respectively). Predictors include health, roster exposure, status conditions, observed move usage, switches, and recent health trends.

Three regularized LightGBM regressors learn:

1. Player 1's future ten-turn team-HP change.
2. Player 2's future ten-turn team-HP change.
3. The change in the difference between their total health.

We apply the fitted models at turn 30 to obtain an **extrapolated**, not actually observed, estimate of resource advantage ten turns later. This is subject to distribution shift: training prefixes stop at turn 20, whereas the final prediction uses turn 30.

## Five-fold results

| Model or ensemble | OOF accuracy | OOF ROC AUC |
| --- | ---: | ---: |
| Original LightGBM/XGBoost/CatBoost (45/45/10) | **69.79%** | 0.76559 |
| Original features + ten-turn forecast, XGBoost | 69.20% | 0.76398 |
| Original features + forecast + transition pressure, XGBoost | 69.38% | 0.76422 |
| Original ensemble 85% + forecast/transition XGBoost 15% | 69.68% | 0.76599 |
| Original ensemble 70% + forecast/transition XGBoost 30% | 69.62% | **0.76617** |

All additional models and weights above were evaluated using the *same* outer fold splits as the original ensemble. Machine-readable verified scores are in `docs/TRANSITION_CV_RESULTS.json`. Probability blending slightly improved the auxiliary metric ROC AUC, but **every tested blend reduced the competition metric, accuracy**. Small variations in these scores should not be overinterpreted; blend weights have been evaluated on the same CV predictions, not on an independent tuning set.

**Decision:** Keep the original 69.79% ensemble and the three existing submission files. This experiment does not justify changing them or claiming that 72–80% accuracy has been achieved.

## Reproduction

Install the existing dependencies from `requirements.txt`, then run:

```bash
python transition_experiment.py --data /path/to/competition/files \
  --out docs/transition_cv.json
```

The directory must contain `train.jsonl`, `test.jsonl`, and `type_chart_gen2.csv`. The script refits the original models and both auxiliary approaches in the same five-fold protocol, prints their scores, and saves a JSON summary. It may require several minutes on a CPU environment.

For a quicker experiment that does not refit the original three classifiers, use:

```bash
python transition_experiment.py --data /path/to/competition/files \
  --skip-original --out docs/transition_cv.json
```

All code, docstrings, comments, and published research material are written in English. The **original** three Kaggle submissions can still be reproduced using `python run.py --data ... --out submissions`.

## Limitations and possible next directions

- Health change is not equivalent to clean inflicted damage; passive effects and move failures act as nuisance variables.
- The estimator only evaluates observed moves and revealed opposing Pokémon. The hidden part of the opposing lineup therefore remains a source of uncertainty.
- Using earlier prefixes to forecast beyond turn 30 requires extrapolation into states not represented by the training snapshots.
- The auxiliary feature set is highly correlated with the original HP, type-matchup, and move features, so it provides limited independent winner information.
- Alternative directions include learning move-selection probabilities, modeling team composition jointly with the hidden roster, and more rigorous grouped/repeated validation to test whether apparent gains survive uncertainty.

Neither the leaderboard nor hidden test outcomes were used to select these models.
