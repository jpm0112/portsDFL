#!/bin/bash
# Shared ASAX job preamble — sourced by hpc/train_model.pbs (not run directly).
# Loads the module stack, activates the slim conda env, cd's to the repo, and
# points PORTSDFL_DATA at the committed dataset. No `set -e`: a benign `module`
# warning must not kill the job; real failures below exit explicitly.
set -uo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$HOME/portsDFL}"
cd "$REPO_ROOT" || { echo "cannot cd to $REPO_ROOT" >&2; exit 1; }

# ASAX uses a dynamic Lmod profile; source it before any `module` call.
source /apps/profiles/modules_asax.sh.dyn
module load anaconda/3-2025.12
module load cuda/11.8.0

# Activate the slim prediction env (create once: conda env create -f hpc/environment-predict.yml).
source activate portsdfl-predict 2>/dev/null || conda activate portsdfl-predict || {
  echo "Failed to activate conda env 'portsdfl-predict'. Create it once with:" >&2
  echo "  conda env create -f $REPO_ROOT/hpc/environment-predict.yml" >&2
  exit 1
}

# Force the env's libs ahead of the module defaults (works around ASAX's setup).
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Data is committed, so default to the repo-root CSV; export PORTSDFL_DATA before
# qsub to point at a different location on the cluster.
export PORTSDFL_DATA="${PORTSDFL_DATA:-$REPO_ROOT/data/training_dataset.csv}"
