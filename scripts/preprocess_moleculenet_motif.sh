#!/bin/bash --login
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J preprocess-moleculenet-motif
#SBATCH -o preprocess-moleculenet-motif.%J.out
#SBATCH -e preprocess-moleculenet-motif.%J.err
#SBATCH --time=12:00:00
#SBATCH --mem=250G

set -euo pipefail

# Adjust to your conda env
conda activate /ibex/user/songt/conda_envs/momo

DATA_ROOT=${DATA_ROOT:-/ibex/user/songt/datasets/dataset}
LIMIT=${LIMIT:--1}

# You can override DATASETS via env, e.g.:
#   DATASETS="bbbp,esol,freesolv,lipophilicity,tox21,toxcast,clintox,sider,muv,hiv,bace"
DATASETS=${DATASETS:-"bbbp,esol,freesolv,lipophilicity,tox21,toxcast,clintox,sider,muv,hiv,bace"}
ATTACH_Y=${ATTACH_Y:-1}

echo "DATA_ROOT=${DATA_ROOT}"
echo "DATASETS=${DATASETS}"
echo "LIMIT=${LIMIT}"
echo "ATTACH_Y=${ATTACH_Y}"

# 统一使用新版预处理入口，生成与预训练一致的字段（motif 图 + 完整目标）
/ibex/user/songt/conda_envs/momo/bin/python -m momo.tools.preprocess_pcqm4mv2 \
  --molnet-root "${DATA_ROOT}" \
  --molnet-list "${DATASETS}" \
  --limit "${LIMIT}" \
  $( [[ "$ATTACH_Y" == "1" ]] && echo "--molnet-attach-y" )
