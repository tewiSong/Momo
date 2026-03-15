#!/bin/bash --login
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J gen-moleculenet-splits
#SBATCH -o gen-moleculenet-splits.%J.out
#SBATCH -e gen-moleculenet-splits.%J.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G

set -euo pipefail

conda activate /ibex/user/songt/conda_envs/momo

DATA_ROOT=${DATA_ROOT:-/ibex/user/songt/datasets/dataset}
DATASETS=${DATASETS:-"bbbp,esol,freesolv,lipophilicity,tox21,toxcast,clintox,sider,muv,hiv,bace"}
MODE=${MODE:-scaffold}   # scaffold | random | random_scaffold
SEED=${SEED:-0}

echo "DATA_ROOT=${DATA_ROOT}"
echo "DATASETS=${DATASETS}"
echo "MODE=${MODE}"
echo "SEED=${SEED}"

/ibex/user/songt/conda_envs/momo/bin/python -m momo.tools.gen_moleculenet_splits \
  --root "${DATA_ROOT}" \
  --list "${DATASETS}" \
  --mode "${MODE}" \
  --seed "${SEED}"

