#!/bin/bash --login
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J preprocess-pcqm4mv2
#SBATCH -o preprocess-pcqm4mv2.%J.out
#SBATCH -e preprocess-pcqm4mv2.%J.err
#SBATCH --time=12:00:00
#SBATCH --mem=250G

set -euo pipefail

conda activate /ibex/user/songt/conda_envs/momo

# ---------------- PCQM4Mv2 (SDF) ----------------
# Default: only run PCQM here. Enable MoleculeNet explicitly if needed.
RUN_PCQM=${RUN_PCQM:-1}
PCQM_SDF=${PCQM_SDF:-data/pcqm4m-v2-train.sdf}
PCQM_OUT=${PCQM_OUT:-data/pcqm4m_v2_motif_preprocessed.pkl}
PCQM_LIMIT=${PCQM_LIMIT:--1}

if [[ "$RUN_PCQM" == "1" ]]; then
  echo "[PCQM] sdf=$PCQM_SDF out=$PCQM_OUT limit=$PCQM_LIMIT"
  /ibex/user/songt/conda_envs/momo/bin/python -m momo.tools.preprocess_pcqm4mv2 \
    --sdf "$PCQM_SDF" \
    --out "$PCQM_OUT" \
    --limit "$PCQM_LIMIT"
fi

# ---------------- MoleculeNet (SMILES) ----------------
# Do not run MoleculeNet by default from this script.
# Use scripts/preprocess_moleculenet_motif.sh or set RUN_MOLNET=1 explicitly.
RUN_MOLNET=${RUN_MOLNET:-0}
DATA_ROOT=${DATA_ROOT:-/ibex/user/songt/datasets/dataset}
DATASETS=${DATASETS:-"bbbp,esol,freesolv,lipophilicity,tox21,toxcast,clintox,sider,muv,hiv,bace"}
MOLNET_LIMIT=${MOLNET_LIMIT:--1}
ATTACH_Y=${ATTACH_Y:-1}

GEN_SPLITS=${GEN_SPLITS:-1}
SPLIT_MODE=${SPLIT_MODE:-scaffold}  # scaffold|random|random_scaffold
SPLIT_SEED=${SPLIT_SEED:-0}

if [[ "$RUN_MOLNET" == "1" ]]; then
  echo "[MoleculeNet] root=$DATA_ROOT datasets=$DATASETS limit=$MOLNET_LIMIT attach_y=$ATTACH_Y"
  /ibex/user/songt/conda_envs/momo/bin/python -m momo.tools.preprocess_pcqm4mv2 \
    --molnet-root "$DATA_ROOT" \
    --molnet-list "$DATASETS" \
    --limit "$MOLNET_LIMIT" \
    $( [[ "$ATTACH_Y" == "1" ]] && echo "--molnet-attach-y" )

  if [[ "$GEN_SPLITS" == "1" ]]; then
    echo "[MoleculeNet Splits] mode=$SPLIT_MODE seed=$SPLIT_SEED"
    /ibex/user/songt/conda_envs/momo/bin/python -m momo.tools.preprocess_pcqm4mv2 \
      --molnet-root "$DATA_ROOT" \
      --molnet-list "$DATASETS" \
      --gen-splits \
      --split-mode "$SPLIT_MODE" \
      --split-seed "$SPLIT_SEED"
  fi
fi
