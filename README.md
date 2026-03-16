# 使用说明

- 激活环境
  - `conda activate /ibex/user/songt/conda_envs/momo`

- 安装依赖
  - `/ibex/user/songt/conda_envs/momo/bin/pip install -r requirements.txt`
  - `/ibex/user/songt/conda_envs/momo/bin/pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.1+cu121.html`
  - `/ibex/user/songt/conda_envs/momo/bin/pip install torch-geometric==2.5.3`
  - `/ibex/user/songt/conda_envs/momo/bin/pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html`
  - `/ibex/user/songt/conda_envs/momo/bin/pip install torch-geometric==2.7.0`

- 数据预处理（PCQM4Mv2 SDF → 预处理 pkl）
  - `/ibex/user/songt/conda_envs/momo/bin/python -m momo.tools.preprocess_pcqm4mv2 --sdf data/pcqm4m-v2-train.sdf --out data/pcqm4m_v2_motif_preprocessed.pkl --limit -1`

- 前向与损失自测
  - `/ibex/user/songt/conda_envs/momo/bin/python -m momo.tools.sanity_pretrain_forward --cfg configs/pretrain_pcqm4mv2.yaml`

- 预训练（含验证与 TensorBoard）
  - `python -m momo.train.pretrain --cfg configs/pretrain_pcqm4mv2.yaml --run_name pretrain-local`

- 查看训练曲线
  - `tensorboard --logdir runs`

- 断点恢复
  - `python -m momo.train.pretrain --cfg configs/pretrain_pcqm4mv2.yaml --run_name pretrain-local --resume checkpoints/pretrain-local/last.ckpt`

- 预训练
  - `sbatch scripts/submit_pretrain.sh`

- 微调数据预处理（MoleculeNet 批量）
  - `sbatch scripts/preprocess_moleculenet_motif.sh`

- 生成划分（MoleculeNet，scaffold）
  - `sbatch scripts/gen_moleculenet_splits.sh`

- 微调与测试（MoleculeNet 批量）
  - `sbatch scripts/finetune_moleculenet.sh`
  - 切换数据集：`DATASETS="bbbp,tox21" sbatch scripts/finetune_moleculenet.sh`（逗号分隔支持批量）
  - 切换 checkpoint：`CKPT=checkpoints/<your-pretrain>/best.ckpt sbatch scripts/finetune_moleculenet.sh`
  - 切换数据根与划分：`ROOT=data/MoleculeNet MODE=scaffold SEED=0 sbatch scripts/finetune_moleculenet.sh`
  - 示例（在 bbbp 上用指定 checkpoint 微调）：

```
ROOT=data/MoleculeNet \
CKPT=checkpoints/pretrain-pcqm4mv2-20260310-180746-gpu214-02/best.ckpt \
DATASETS="bbbp" MODE=scaffold SEED=0 EPOCHS=100 BATCH=256 LR=1e-3 LR_SCALE=5 POOL=sum \
sbatch scripts/finetune_moleculenet.sh
```
