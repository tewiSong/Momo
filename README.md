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

- Slurm 提交
  - `sbatch scripts/submit_pretrain.sh`

- 微调数据预处理（MoleculeNet 批量）
  - `sbatch scripts/preprocess_moleculenet_motif.sh`

- 生成划分（MoleculeNet，scaffold）
  - `sbatch scripts/gen_moleculenet_splits.sh`
