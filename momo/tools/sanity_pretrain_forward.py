#!/usr/bin/env python
import argparse
import torch
from torch_geometric.loader import DataLoader

from momo.utils.config import load_yaml
from momo.data.pcqm4mv2 import PCQM4Mv2MotifDataset, motif_global_index
from momo.models.gin_motif_vqmoe import MotifVQMoE
from momo.train.losses import compute_losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', type=str, required=True)
    args = p.parse_args()

    cfg = load_yaml(args.cfg)
    device = cfg['misc']['device']

    ds = PCQM4Mv2MotifDataset(
        preprocessed_path=cfg['dataset']['preprocessed_path'],
        z3d_dim=cfg['model']['z3d_dim'],
        max_atomic_num=cfg['dataset']['max_atomic_num'],
        require_pos=bool(cfg.get('teacher', {}).get('enabled', False)),
    )
    dl = DataLoader(ds,
                    batch_size=cfg['dataset']['batch_size'],
                    shuffle=cfg['dataset']['shuffle'],
                    num_workers=cfg['dataset']['num_workers'])

    model = MotifVQMoE(cfg).to(device)

    batch = next(iter(dl))
    batch = batch.to(device)

    z_hat, h_motif, e_k, logits, topk, router_info = model(batch)
    # 与 GT 对齐
    z_gt = batch.motif_target
    assert z_gt.shape == z_hat.shape

    losses = compute_losses(
        z_hat=z_hat,
        z_gt=z_gt,
        h_motif_2d=h_motif,
        e_k=e_k,
        weights=cfg['loss'],
        router=router_info,
        teacher_z=router_info.get('teacher_z') if router_info is not None else None,
    )
    print({k: float(v.detach().cpu()) for k, v in losses.items()})


if __name__ == '__main__':
    main()
