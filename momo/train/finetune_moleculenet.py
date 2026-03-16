#!/usr/bin/env python
"""
Finetune Momo (MotifVQMoE backbone) on MoleculeNet datasets with GraphMVP-style evaluation.

Usage example:
  python -m momo.train.finetune_moleculenet \
    --root data/MoleculeNet --dataset bbbp \
    --preproc-name bbbp_motif_preprocessed.pkl \
    --mode scaffold --seed 0 \
    --ckpt <path/to/pretrained_momo.ckpt>  (optional) \
    --epochs 100 --batch-size 256 --lr 1e-3 --lr-scale 5.0

Notes:
  - Run preprocessing first:
      python momo/tools/preprocess_moleculenet_motif.py --root GraphMVP/datasets/molecule_datasets --dataset bbbp --attach-y
  - Generate splits first:
      python momo/tools/gen_moleculenet_splits.py --root GraphMVP/datasets/molecule_datasets --dataset bbbp --mode scaffold
  - Then copy/alias that dataset folder into data/MoleculeNet/bbbp (or point --root to GraphMVP/...)
"""

import os
import argparse
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from momo.data.moleculenet import MoleculeNetMotifDataset, load_split_indices, CLASSIFICATION_DATASETS, REGRESSION_DATASETS
from momo.data.pcqm4mv2 import motif_global_index, _infer_num_motifs_per_graph
from momo.models.gin_motif_vqmoe import MotifVQMoE
from momo.utils.config import load_yaml
from rdkit.Chem.Scaffolds import MurckoScaffold


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(wanted: str = 'cuda') -> str:
    if wanted == 'cuda':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return wanted


def build_model(cfg_path: str, ckpt: str | None, device: str) -> MotifVQMoE:
    cfg = load_yaml(cfg_path)
    model = MotifVQMoE(cfg).to(device)
    # 下游 MoleculeNet 只用 2D 拓扑，禁用 teacher 分支以避免需要 pos
    if hasattr(model, 'teacher_enabled'):
        model.teacher_enabled = False
    if ckpt:
        state = torch.load(ckpt, map_location=device)
        if 'model' in state:
            model.load_state_dict(state['model'], strict=False)
        else:
            model.load_state_dict(state, strict=False)
    return model


def motif_pool_to_graph(feat: torch.Tensor, motif_id: torch.Tensor, batch: torch.Tensor, pooling: str = 'sum') -> torch.Tensor:
    """将 motif 级特征聚合到图级（sum/mean 读出）。

    feat: [N_motifs_total, D]
    motif_id: [N_nodes] 每原子 motif 的局部 id (0..M_g-1)
    batch: [N_nodes] 每原子所属图索引 (0..B-1)
    """
    g_counts = _infer_num_motifs_per_graph(motif_id, batch)  # [B]
    offsets = torch.cumsum(torch.cat([torch.zeros(1, device=g_counts.device, dtype=g_counts.dtype), g_counts], dim=0), dim=0)
    # motif -> graph id
    motif_graph_ids = torch.repeat_interleave(torch.arange(g_counts.numel(), device=g_counts.device), g_counts)
    assert motif_graph_ids.numel() == feat.size(0)
    out = torch.zeros((g_counts.numel(), feat.size(1)), dtype=feat.dtype, device=feat.device)
    out = out.index_add(0, motif_graph_ids, feat)
    if pooling == 'mean':
        denom = g_counts.clamp(min=1).view(-1, 1).to(out.dtype)
        out = out / denom
    return out


def forward_graph_repr(model: MotifVQMoE, batch: 'torch_geometric.data.Batch', pooling: str = 'sum') -> torch.Tensor:
    # 模型前向：返回 z_hat(=E_k+Δ), h_commit(2D->z3d投影)
    z_hat, h_commit, e_k, logits, topk, router_info = model(batch)
    # 按 doc.md：使用增强特征 [h_commit, z_hat] 进行下游聚合
    motif_feat = torch.cat([h_commit, z_hat], dim=-1)
    g_repr = motif_pool_to_graph(motif_feat, batch.motif_id, batch.batch, pooling=pooling)
    return g_repr


def build_head(in_dim: int, num_tasks: int) -> nn.Module:
    return nn.Linear(in_dim, num_tasks)


def train_one_epoch(model: MotifVQMoE, head: nn.Module, dl: DataLoader, optim: torch.optim.Optimizer,
                    criterion: nn.Module, device: str, task: str, pooling: str) -> float:
    model.train()
    head.train()
    total = 0.0
    for batch in dl:
        batch = batch.to(device)
        g_repr = forward_graph_repr(model, batch, pooling=pooling)
        pred = head(g_repr)
        if task == 'clf':
            y = batch.y.view(pred.shape).to(torch.float32)
            is_valid = (y ** 2) > 0  # -1/1 掩码
            loss_mat = criterion(pred, (y + 1.0) / 2.0)
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros_like(loss_mat))
            loss = torch.sum(loss_mat) / torch.clamp(is_valid.sum(), min=1)
        else:
            y = batch.y.view(pred.shape).to(torch.float32)
            loss = criterion(pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total += float(loss.detach().item())
    return total / max(1, len(dl))


def eval_epoch(model: MotifVQMoE, head: nn.Module, dl: DataLoader, device: str, task: str, pooling: str) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    head.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in dl:
            batch = batch.to(device)
            g_repr = forward_graph_repr(model, batch, pooling=pooling)
            pred = head(g_repr)
            ys.append(batch.y.view(pred.shape).cpu())
            ps.append(pred.cpu())
    y = torch.cat(ys, dim=0).numpy()
    p = torch.cat(ps, dim=0).numpy()
    if task == 'clf':
        from sklearn.metrics import roc_auc_score
        roc_list = []
        for i in range(y.shape[1]):
            if (y[:, i] == 1).sum() > 0 and (y[:, i] == -1).sum() > 0:
                msk = (y[:, i] ** 2) > 0
                roc_list.append(roc_auc_score((y[msk, i] + 1) / 2.0, p[msk, i]))
        return {'ROC-AUC': float(np.mean(roc_list)) if roc_list else 0.0}, y, p
    else:
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        rmse = mean_squared_error(y.reshape(-1), p.reshape(-1), squared=False)
        mae = mean_absolute_error(y.reshape(-1), p.reshape(-1))
        return {'RMSE': float(rmse), 'MAE': float(mae)}, y, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, default='data/MoleculeNet', help='MoleculeNet 根目录（含各 dataset 子目录）')
    ap.add_argument('--dataset', type=str, required=True, help='数据集名，例如 bbbp, tox21, esol 等')
    ap.add_argument('--preproc-name', type=str, default='', help='预处理 pkl 文件名，留空则使用 {dataset}_motif_preprocessed.pkl')
    ap.add_argument('--mode', type=str, default='scaffold', choices=['scaffold', 'random', 'random_scaffold'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cfg', type=str, default='configs/momo.yaml', help='模型配置（与预训练同结构）')
    ap.add_argument('--ckpt', type=str, default='', help='可选：加载预训练权重 ckpt（包含 model 或纯 state_dict）')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr-scale', type=float, default=5.0, help='head 层学习率缩放')
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--save-dir', type=str, default='runs')
    ap.add_argument('--graph-pooling', type=str, default='sum', choices=['sum', 'mean'])
    ap.add_argument('--freeze-codebook', action='store_true', help='微调时冻结 codebook（推荐）')
    ap.add_argument('--freeze-backbone', action='store_true', help='可选：冻结 2D 主干，仅训练 head（不推荐）')
    ap.add_argument('--ft-cfg', type=str, default='', help='可选：微调 YAML（提供默认与 per-dataset 覆盖）')
    args = ap.parse_args()

    setup_seed(args.seed)
    device = get_device(args.device)

    ds_dir = os.path.join(args.root, args.dataset)
    pkl_name = args.preproc_name or f"{args.dataset}_motif_preprocessed.pkl"
    pkl_path = os.path.join(ds_dir, pkl_name)
    assert os.path.exists(pkl_path), f"预处理文件不存在: {pkl_path}"

    # 可选：从 YAML 读取微调参数覆盖 CLI（更便于管理不同数据集的差异）
    if args.ft_cfg and os.path.exists(args.ft_cfg):
        ft = load_yaml(args.ft_cfg)
        defaults = ft.get('defaults', {})
        specific = ft.get('per_dataset', {}).get(args.dataset, {})
        merged = {}
        merged.update(defaults)
        merged.update(specific)
        # 允许覆盖的键
        key_map = {
            'epochs': 'epochs',
            'batch_size': 'batch_size',
            'lr': 'lr',
            'lr_scale': 'lr_scale',
            'graph_pooling': 'graph_pooling',
            'freeze_codebook': 'freeze_codebook',
            'freeze_backbone': 'freeze_backbone',
            'mode': 'mode',
            'seed': 'seed',
        }
        for k_src, k_dst in key_map.items():
            if k_src in merged:
                setattr(args, k_dst, merged[k_src])

    # 如果 splits 缺失，动态生成并固化
    def _read_smiles(path: str) -> list[str]:
        smiles_list: list[str] = []
        with open(path, 'r') as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if ',' in s:
                    s = s.split(',')[0]
                smiles_list.append(s)
        return smiles_list

    def _random_split_indices(n: int, frac_train: float, frac_valid: float, frac_test: float, seed: int):
        rng = np.random.RandomState(seed)
        all_idx = np.arange(n)
        rng.shuffle(all_idx)
        n_train = int(frac_train * n)
        n_valid = int(frac_valid * n)
        train_idx = all_idx[:n_train].tolist()
        valid_idx = all_idx[n_train:n_train + n_valid].tolist()
        test_idx = all_idx[n_train + n_valid:].tolist()
        return train_idx, valid_idx, test_idx

    def _scaffold_smiles(smiles: str) -> str:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=True)

    def _scaffold_split_indices(smiles_list: list[str], frac_train: float, frac_valid: float, frac_test: float):
        from collections import defaultdict
        # group by scaffold
        sc2idx = {}
        for i, smi in enumerate(smiles_list):
            sc = _scaffold_smiles(smi)
            sc2idx.setdefault(sc, []).append(i)
        all_sets = [sorted(v) for (k, v) in sorted(sc2idx.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)]
        n = len(smiles_list)
        train_cut = frac_train * n
        valid_cut = (frac_train + frac_valid) * n
        tr, va, te = [], [], []
        for sset in all_sets:
            if len(tr) + len(sset) > train_cut:
                if len(tr) + len(va) + len(sset) > valid_cut:
                    te.extend(sset)
                else:
                    va.extend(sset)
            else:
                tr.extend(sset)
        return tr, va, te

    def _random_scaffold_split_indices(smiles_list: list[str], frac_train: float, frac_valid: float, frac_test: float, seed: int):
        from collections import defaultdict
        scaffolds = defaultdict(list)
        for i, smi in enumerate(smiles_list):
            scaffolds[_scaffold_smiles(smi)].append(i)
        rng = np.random.RandomState(seed)
        sc_sets = list(scaffolds.values())
        rng.shuffle(sc_sets)
        n_total = len(smiles_list)
        n_valid = int(np.floor(frac_valid * n_total))
        n_test = int(np.floor(frac_test * n_total))
        tr, va, te = [], [], []
        for sset in sc_sets:
            if len(va) + len(sset) <= n_valid:
                va.extend(sset)
            elif len(te) + len(sset) <= n_test:
                te.extend(sset)
            else:
                tr.extend(sset)
        return tr, va, te

    def ensure_splits(ds_dir: str, mode: str, seed: int, smiles_from_pkl: list[str] | None) -> tuple[list[int], list[int], list[int]]:
        try:
            return load_split_indices(ds_dir, mode=mode, seed=seed)
        except Exception:
            pass
        # 构建并固化
        smi_csv = os.path.join(ds_dir, 'processed', 'smiles.csv')
        if os.path.exists(smi_csv):
            smiles = _read_smiles(smi_csv)
        else:
            if smiles_from_pkl is None or len(smiles_from_pkl) == 0:
                raise FileNotFoundError(f"Missing splits and no smiles.csv; cannot fallback. Looked for {smi_csv}")
            smiles = smiles_from_pkl
        if mode == 'scaffold':
            tr, va, te = _scaffold_split_indices(smiles, 0.8, 0.1, 0.1)
            out_base = os.path.join(ds_dir, 'splits', mode)
        elif mode == 'random_scaffold':
            tr, va, te = _random_scaffold_split_indices(smiles, 0.8, 0.1, 0.1, seed)
            out_base = os.path.join(ds_dir, 'splits', f'{mode}_seed{seed}')
        else:
            tr, va, te = _random_split_indices(len(smiles), 0.8, 0.1, 0.1, seed)
            out_base = os.path.join(ds_dir, 'splits', f'{mode}_seed{seed}')
        os.makedirs(out_base, exist_ok=True)
        for name, idx in [('train.txt', tr), ('val.txt', va), ('test.txt', te)]:
            with open(os.path.join(out_base, name), 'w') as f:
                for i in idx:
                    f.write(str(int(i)) + '\n')
        return tr, va, te

    # 先构造数据集（需要 pkl，且如果缺 smiles.csv 时可从 pkl 取得 smiles）
    is_clf = args.dataset in CLASSIFICATION_DATASETS
    is_reg = args.dataset in REGRESSION_DATASETS
    ds = MoleculeNetMotifDataset(preprocessed_path=pkl_path, dataset_name=args.dataset, require_pos=False)
    smiles_from_pkl = [rec.get('smiles', '') for rec in ds.data] if hasattr(ds, 'data') else None

    # 读取或生成划分索引
    tr_idx, va_idx, te_idx = ensure_splits(ds_dir, mode=args.mode, seed=args.seed, smiles_from_pkl=smiles_from_pkl)
    # 将基于 smiles.csv 的索引映射到 pkl 子集（过滤预处理失败的样本），避免越界
    tr_idx = ds.remap_indices(tr_idx)
    va_idx = ds.remap_indices(va_idx)
    te_idx = ds.remap_indices(te_idx)
    sub = torch.utils.data.Subset
    ds_tr, ds_va, ds_te = sub(ds, tr_idx), sub(ds, va_idx), sub(ds, te_idx)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 模型与 head
    model = build_model(args.cfg, args.ckpt or None, device)
    # 微调策略：冻结 codebook（与 doc.md 一致）
    if args.freeze_codebook and hasattr(model, 'codebook'):
        for p in model.codebook.parameters():
            p.requires_grad = False
    if args.freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    # 预测头输入为 [h_commit, z_hat] 拼接 => 2 * z3d_dim
    head_in = int(model.z3d_dim) * 2
    head = build_head(head_in, ds.num_tasks).to(device)

    # 优化器（与 GraphMVP 一致：骨干和 head 分组，不同 lr）
    params = [
        {'params': model.parameters(), 'lr': args.lr},
        {'params': head.parameters(), 'lr': args.lr * args.lr_scale},
    ]
    optim = torch.optim.Adam(params, lr=args.lr)
    if is_clf:
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        task = 'clf'
    else:
        criterion = nn.MSELoss()
        task = 'reg'

    best_metric = -1e18 if is_clf else 1e18
    best = None
    for ep in range(1, args.epochs + 1):
        loss = train_one_epoch(model, head, dl_tr, optim, criterion, device, task, pooling=args.graph_pooling)
        val_metric, yv, pv = eval_epoch(model, head, dl_va, device, task, pooling=args.graph_pooling)
        te_metric, yt, pt = eval_epoch(model, head, dl_te, device, task, pooling=args.graph_pooling)
        if is_clf:
            cur = val_metric['ROC-AUC']
            better = cur > best_metric
        else:
            cur = val_metric['RMSE']
            better = cur < best_metric
        if better:
            best_metric = cur
            best = (val_metric, te_metric, yt, pt)
        print(f"Epoch {ep} loss={loss:.4f}  val={val_metric}  test={te_metric}")

    # 输出结果到数据目录
    out_dir = os.path.join(ds_dir, 'momo_results')
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{args.mode}_seed{args.seed}"
    out_npz = os.path.join(out_dir, f"{args.dataset}_{tag}.npz")
    if best is not None:
        val_metric, te_metric, yt, pt = best
    else:
        _, _, yt, pt = eval_epoch(model, head, dl_te, device, task)
        val_metric, te_metric, = {}, {}
    np.savez(out_npz, test_target=yt, test_pred=pt, val_metric=val_metric, test_metric=te_metric)
    print(f"Saved results to: {out_npz}")


if __name__ == '__main__':
    main()
