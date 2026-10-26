# Pokémon Battle Prediction — FDS Kaggle Competition

**Task:** predict whether Player 1 eventually wins a Gen 2 OU battle, using only Player 1's team, the opposing lead, and the first 30 observed turns. The target is `player_won` (0 or 1).

The repository contains an executable **classical machine learning** pipeline using only the official competition inputs. Neural networks, external Pokémon databases, extra replay files, and pretrained models are not used.

## Cross-validated results

The most recent **user-reported Kaggle run (26 October 2026)** generated
10,000 out-of-fold predictions using stratified five-fold cross-validation
(seed 42) and leakage-safe nested team target encoding. Its reported direct
dependency versions are pinned in [`requirements.txt`](requirements.txt).
Scores below are the **measured training OOF accuracies**, not Kaggle test scores.

| Model or submission | Models / blend weights | Kaggle OOF accuracy |
| --- | --- | ---: |
| LightGBM | LGBM | 69.33% |
| XGBoost | XGB | 69.47% |
| CatBoost | CAT | 69.31% |
| `sub1_lgbm.csv` | LGBM 100% | 69.33% |
| `sub2_lgb_xgb.csv` | LGBM 50% + XGB 50% | 69.64% |
| **`sub3_lgb_xgb_cat.csv`** | **LGBM 45% + XGB 45% + CAT 10%** | **69.72%** |

Earlier experiments in a different environment reported LightGBM at 69.50%,
XGBoost at 69.47%, CatBoost at 69.31%, the two-model blend at 69.70% and
the three-model blend at 69.79%. These historical scores remain in the
research reports for transparency, but **69.72% is the most recent Kaggle
baseline reported for the pinned submission setup**. Version pinning improves
repeatability but does not guarantee bit-for-bit identical results across
platforms or installations.

The notebook prints the **actual measured scores** and writes
`run_manifest.json` containing dependency versions, the Git commit, input
checksums, feature columns, model parameters and cross-validation settings.
See [`docs/KAGGLE_REPRODUCIBILITY.md`](docs/KAGGLE_REPRODUCIBILITY.md).
`requirements-kaggle.txt` is kept for a **separate historical comparison**
using XGBoost 3.4.1 and CatBoost 1.2.10; it is not used by the submission
notebook.

**The 80% target has not been achieved.** The recent three-model blend
improves on LightGBM alone by 0.39 percentage points on the same run.
Blend selection on training OOF results can be optimistic; private
test performance is unknown.

## Repository structure

```text
pokemon/
  __init__.py
  features.py       Static team statistics, active state, move usage
  features2.py      Remaining resources, healing, status, KO timing
  features3.py      Switches and approximate missing KO observations
  features4.py      Candidate lineups and type matchups
  features5.py      Experimental temporal features (not in final pipeline)
  transitions.py    Self-supervised HP transition and attack-pressure features
  forecast.py       Self-supervised ten-turn team resource forecasting
  te.py             Leakage-safe, nested team target encoding
  main.py           Data processing, models, cross-validation, submissions
notebooks/
  kaggle_notebook.ipynb  Public-notebook entry point, clones this repository
submissions/
  sub1_lgbm.csv
  sub2_lgb_xgb.csv
  sub3_lgb_xgb_cat.csv
transition_experiment.py   Reproducible transition/forecast five-fold research
tests/                  Label-independence and alignment smoke tests
docs/TRANSITION_EXPERIMENT.md     Research methods and interpretation
docs/TRANSITION_CV_RESULTS.json Five-fold observed experiment scores
run.py                  Command-line interface
requirements.txt
README.md
```

## Reproduce the experiments

1. Install the dependencies: `python -m pip install -r requirements.txt`.
2. Place the **original** `train.jsonl`, `test.jsonl`, and `type_chart_gen2.csv` in a directory. They are **not** committed to the repository.
3. Rebuild all submissions using:

```bash
python run.py --data /path/to/data --out submissions
```

4. Reproduce the outer five-fold validation scores using:

```bash
python run.py --data /path/to/data --out submissions --cv
```

Run `notebooks/kaggle_notebook.ipynb` on Kaggle. The clone cell already points to this public repository and installs pinned packages from `requirements.txt`. Enable Kaggle Internet access for cloning and dependency installation, and restart the kernel if it has loaded different versions. The notebook refreshes an existing checkout using a fast-forward-only pull.

When explicitly enabled, the notebook also evaluates the 42-feature active-strength candidate with seed
42 and saves its OOF predictions, scores and separate candidate CSVs. The optional experiment is **disabled by default** (`RUN_STRENGTH_RESEARCH = False`). Set it to `True` only when separately evaluating the candidate. A follow-up Kaggle run measured 69.64%, compared with 69.85%
locally; this candidate is not part of the original 637-feature baseline.

The three submissions must preserve the official template's `battle_id` order and contain binary predictions. Regenerate CSVs from the current notebook before submitting: files already committed under `submissions/` are previous snapshots, not automatically updated by a dependency change. Follow the competition's daily limits and final selection requirements.

## Feature engineering

**Static setup:** P1 party statistics and items, the P2 lead's base stats, and type effectiveness from the official Gen 2 chart.

**Observed battle state:** active Pokémon HP, statuses, boosts and volatile effects; repeated switches; damage estimates; HP advantage trends; move usage and party identities.

**Remaining resources:** estimated total party HP, alive/fainted Pokémon stats, healing moves, switches and approximated knockouts when the log does not explicitly report a faint.

**Team priors:** compatible P2 party candidates are matched against team identities found in the supplied data. Team win rates are supervised features, **never** computed on the corresponding outer-validation outcomes.

## Model experiments and limitations

- LightGBM hyperparameter alternatives around the original baseline showed no consistent improvement in previous experiments.
- Extra team-HP time series, per-species HP/status signals and move-pressure features (`features5.py`) were evaluated on the **first** held-out fold. They did not improve the baseline on that split (baseline **69.10%**, best of the tested temporal configurations **68.95%**). These features are kept as documented experiments, not included in the final submission pipeline.
- Building the candidate-team pool from both known P1 lineups and fully revealed P2 lineups reduces cases without candidates, but on the first fold accuracy decreased from **69.10%** to **68.80%**. The original candidate-pool approach is preserved.
- A regularized linear player-team rating model based on individual species could not approach the boosted models on the first fold (about 56% accuracy).
- All classical base models have very similar validation scores. Small ensemble gains should not be interpreted as evidence that the 80% target is attainable.

## State-transition and resource-forecasting research

Two new classical auxiliary learners were implemented and evaluated with the
same five-fold protocol: a net-HP-change regressor based on observed turns,
and a ten-turn team-resource forecast trained on earlier prefixes. They were
combined with the original features using XGBoost. **Neither improved the
original 69.79% accuracy**. The strongest experimental classifier reached
69.38%, and its 30% probability blend with the original ensemble reached
69.62% (but marginally improved ROC AUC).

Full English-language methodology, model limitations, and results are in
[`docs/TRANSITION_EXPERIMENT.md`](docs/TRANSITION_EXPERIMENT.md).
The experiment is completely separate from the three selected submissions:

```bash
python transition_experiment.py --data /path/to/data --out docs/transition_cv.json
```

You may optionally provide all three existing feature caches to avoid rebuilding
expensive input matrices (`--baseline-cache`, `--transition-cache`, and
`--forecast-cache`). To skip refitting the original models, add
`--skip-original`. To run the smoke tests:

```bash
POKEMON_DATA_DIR=/path/to/data python -m unittest discover -s tests -v
```

## Competition rules and submission checklist

- Use only the original train/test JSONL files and official Generation 2 type chart.
- Avoid using the test set or leaderboard to select model hyperparameters.
- Select **three** Kaggle submissions before the competition closes on **5 November 2026 at 23:59**.
- After the competition closes, make the Kaggle notebook public. It must **clone the GitHub repository** and reproduce the three selected submissions, with a Markdown explanation before each one.
- Submit the notebook and report by **12 November 2026**.

This project emphasizes methodological transparency and reproducibility. No private leaderboard score is claimed.

## Additional research: player action prediction

A supplementary experiment learns the probability of an offensive move, a
status move, or an unrecorded action from causal, observed prefixes. The
next-action LightGBM classifier achieves **62.34%** five-fold action accuracy
versus a **50.29%** majority-action baseline. However, adding its predictions
to the battle winner classifier does not yield a meaningful gain: the best
15%-weighted blend scores **69.80%**, only one additional correct battle out
of 10,000 compared with the established **69.79%** model. Action-enhanced
XGBoost reaches **69.38%** and its blends perform worse than the baseline.

All code, method details, validation tables, and interpretation are in
[`docs/ACTION_EXPERIMENT.md`](docs/ACTION_EXPERIMENT.md). The source files are
`pokemon/actions.py`, `action_experiment.py`, and `tests/test_actions.py`.
The action experiment does not change the three final submissions. To
reproduce from competition data (without precomputed caches):

```bash
python action_experiment.py --data /path/to/data --include-xgb
```

The script supports optional `--baseline-cache`, `--reference-oof`, and
`--action-cache` arguments to speed up repeated experiments. When an original
reference OOF file is supplied, it must be aligned with the original training
file and five validation folds. Run `python -m unittest discover -s tests -v`
to check temporal boundaries, class definitions, and output shape.

## Further attempts to reach 70% accuracy

Further targeted experiments were performed with Pokémon-pair and matchup
target encodings, first/last-active species identities, alternative XGBoost
depth, fold-local importance-based feature selection, and cross-fold threshold
calibration. No alternative exceeded 70% under the original five-fold setup.
The best reference remains 69.79% out-of-fold accuracy. See
[`docs/RESEARCH_70.md`](docs/RESEARCH_70.md) for the exact results, validation
controls, tested hypotheses, and reproducible commands.

`research_70.py` can independently reproduce the added methods without
altering the three official submission models. All source comments and
function docstrings are in English.

### Active-strength follow-up

A further five-fold experiment adds 42 interactions between active HP, base
statistics, stat stages and status conditions. The strongest individual
XGBoost reaches **69.85% locally**; its five-seed average reaches **69.77%**.
The supplied Kaggle follow-up log measures **69.64%** for the individual
candidate against **69.50%** for the same-run reference; its fixed blend
measures 69.51%. The six reported direct package versions are pinned in
`requirements-kaggle.txt` for controlled comparisons.
Repeating the run locally with all six Kaggle direct versions still gives
69.85%, so those version differences alone did not explain the discrepancy
in the controlled local comparison.
The original ensemble scores **69.42% in this new environment**, so these
figures should not be directly compared with the historical 69.79% as if
they came from an identical run. No tested alternative exceeded 70%.

`strength_experiment.py` records the package versions, input checksum,
per-fold scores and OOF probabilities, and can generate separate candidate
submissions. See [`docs/STRENGTH_EXPERIMENT.md`](docs/STRENGTH_EXPERIMENT.md)
for results, limitations and reproduction commands. The three selected
submission files and the notebook remain unchanged by this experiment.
