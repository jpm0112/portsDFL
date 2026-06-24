# Training the prediction models on ASAX

Train + tune every model on the Alabama Supercomputer (ASAX, PBS/Torque) as one
PBS job per model, then reload the saved artifacts anywhere to predict — no
retraining. Pure Predict-then-Optimize; the DFL/optimizer stack is not involved.

## One-time setup (login node)

```bash
git clone <this repo>            # or: git pull   in an existing checkout
cd portsDFL
conda env create -f hpc/environment-predict.yml      # builds env 'portsdfl-predict'
```

The training data (`data/training_dataset.csv`) is committed, so it arrives with
the clone/pull — nothing to copy. (To use a different copy, `export PORTSDFL_DATA=/path/to.csv`
before submitting.)

## Run loop (every time)

```bash
git pull                          # get the latest code on ASAX
bash hpc/submit_batch.sh          # one PBS job per model (xgb lgbm rf linear realmlp tabm node)
qstat -u $USER                    # watch them run in parallel
# ...once all jobs finish:
python prediction_models/scripts/run_baselines.py     # sanity-floor rows for the table
python prediction_models/scripts/compare.py           # -> results/comparison.csv leaderboard
```

Submit a subset with `bash hpc/submit_batch.sh xgb lgbm`, or preview the exact
`qsub` commands without submitting via `bash hpc/submit_batch.sh --dry-run`.

## Outputs

```
artifacts/                       # portable trained models — copy elsewhere & predict
  preprocessor.pkl  <model>.pkl  <model>.meta.json
results/<model>/                 # cv_summary_tuned.csv, trials.csv, best_config.json, oof_predictions.csv
results/comparison.csv           # cross-model leaderboard (MAE/RMSE/R2/MAPE)
results/hpc_logs/<model>.log     # per-job stdout/stderr
```

Predict on new vessels (any machine with the env + `artifacts/`):

```bash
python prediction_models/scripts/predict.py --input vessels.csv --output preds.csv
```

## Tuning knobs

- **Tuning budget** — `train_all.py` uses 40 trials for trees, 20 for neural. Edit
  the `RESOURCES` walltimes in `submit_batch.sh` if you change these.
- **CPU vs GPU queue** — tree jobs default to `-q gpu` (so a fresh checkout runs
  first try) but don't use the GPU. Once you confirm the ASAX CPU-queue name, set
  the tree rows in `submit_batch.sh` to it with `ngpus 0` to free the GPU.
- **GPU type** — no `gpuname` is pinned (schedules on any free GPU). Append
  `:gpuname=ampere` (or `hopper`) to the `select` line in `submit_batch.sh` to pin one.
- **Email** — `export ASAX_EMAIL=you@domain` before submitting for completion mail.

## Note on the committed data

`data/training_dataset.csv` is tracked in git (commit `0ff16af`) so it reaches ASAX
with the push. That puts proprietary port data into git history — keep the
`jpm0112/portsDFL` remote **private**.
