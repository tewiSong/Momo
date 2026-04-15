#!/bin/bash --login
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J pretrain-pcqm4mv2
#SBATCH -o pretrain-pcqm4mv2.%J.out
#SBATCH -e pretrain-pcqm4mv2.%J.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=250G
#SBATCH --constraint=v100|a100

conda activate /ibex/user/songt/conda_envs/momo
export PYTHONUNBUFFERED=1

# Launch pretraining with TensorBoard logging
python -u -m momo.train.pretrain --cfg configs/pretrain_pcqm4mv2.yaml --run_name pretrain-pcqm4mv2
