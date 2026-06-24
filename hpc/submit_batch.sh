#!/bin/bash
# Submit one ASAX PBS training job per model — maximum parallelism (mirrors the
# repro-track per-task pattern). Each job tunes + refits + saves one model's
# artifact; run scripts/compare.py once they all finish to build the leaderboard.
#
# Usage:
#   bash hpc/submit_batch.sh                  # every model
#   bash hpc/submit_batch.sh xgb lgbm         # a subset
#   bash hpc/submit_batch.sh --dry-run        # print the qsub commands, submit nothing
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRAPPER="$REPO_ROOT/hpc/train_model.pbs"
LOGDIR="$REPO_ROOT/prediction_models/results/hpc_logs"
EMAIL="${ASAX_EMAIL:-}"  # export ASAX_EMAIL to get job-completion mail

# Per-model PBS resources: "queue walltime ncpus mem ngpus".
# ponytail: trees are CPU-bound (n_jobs=-1) but default to the gpu queue so a fresh
# checkout runs first try. Once you confirm the ASAX CPU-queue name, switch the tree
# rows' queue to it (and ngpus 0) so a tree job stops idling a GPU.
declare -A RESOURCES=(
  [xgb]="gpu 02:00:00 8 32gb 1"
  [lgbm]="gpu 02:00:00 8 32gb 1"
  [rf]="gpu 03:00:00 8 32gb 1"
  [linear]="gpu 02:00:00 8 32gb 1"
  [realmlp]="gpu 06:00:00 8 48gb 1"
  [tabm]="gpu 06:00:00 8 48gb 1"
  [node]="gpu 08:00:00 8 48gb 1"
)

DRY_RUN=0
MODELS=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) MODELS+=("$arg") ;;
  esac
done
[ ${#MODELS[@]} -eq 0 ] && MODELS=(xgb lgbm rf linear realmlp tabm node)

mkdir -p "$LOGDIR"
submitted=0
for model in "${MODELS[@]}"; do
  spec="${RESOURCES[$model]:-}"
  if [ -z "$spec" ]; then
    echo "Unknown model '$model' (no RESOURCES entry); skipping." >&2
    continue
  fi
  read -r queue walltime ncpus mem ngpus <<< "$spec"
  select="select=1:ncpus=${ncpus}:mpiprocs=${ncpus}:mem=${mem%gb}000mb"
  [ "$ngpus" -gt 0 ] && select="${select}:ngpus=${ngpus}"

  qsub_cmd=(qsub -q "$queue" -N "portsdfl_${model}" -j oe -o "$LOGDIR/${model}.log"
            -r n -l "walltime=${walltime}" -l "$select" -v "MODEL=${model}")
  [ -n "$EMAIL" ] && qsub_cmd+=(-M "$EMAIL" -m ae)
  qsub_cmd+=("$WRAPPER")

  if [ "$DRY_RUN" -eq 1 ]; then
    printf '%q ' "${qsub_cmd[@]}"; echo
  else
    "${qsub_cmd[@]}"
  fi
  submitted=$((submitted + 1))
done

echo
echo "Prepared/submitted ${submitted} job(s). Monitor with:  qstat -u \$USER"
echo "When all finish:  python prediction_models/scripts/compare.py"
