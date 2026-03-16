#!/bin/bash --login
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J momo-ft
#SBATCH -o momo-ft.%J.out
#SBATCH -e momo-ft.%J.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

conda activate /ibex/user/songt/conda_envs/momo

ROOT=${ROOT:-data/MoleculeNet}
DATASETS=${DATASETS:-"bbbp,esol"}
MODE=${MODE:-scaffold}
SEED=${SEED:-0}
EPOCHS=${EPOCHS:-100}
BATCH=${BATCH:-256}
LR=${LR:-1e-3}
LR_SCALE=${LR_SCALE:-5}
POOL=${POOL:-sum}
CFG=${CFG:-configs/pretrain_pcqm4mv2.yaml}
CKPT=${CKPT:-}

IFS=',' read -ra DS <<< "$DATASETS"
for d in "${DS[@]}"; do
  echo "==> Finetune ${d}"
  python -m momo.train.finetune_moleculenet \
    --root "$ROOT" \
    --dataset "$d" \
    --mode "$MODE" \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --lr "$LR" \
    --lr-scale "$LR_SCALE" \
    --graph-pooling "$POOL" \
    --cfg "$CFG" \
    ${CKPT:+--ckpt "$CKPT"} \
    --freeze-codebook
done

