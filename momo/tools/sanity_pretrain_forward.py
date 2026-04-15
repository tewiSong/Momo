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
        z3d_dim=int(cfg['model'].get('edge_target_dim', 7)),
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

    h_motif_2d, h_motif_enh, router_info = model(batch)
    losses = compute_losses(
        h_motif_2d=h_motif_2d,
        router=router_info,
        weights=cfg['loss'],
    )
    print({k: float(v.detach().cpu()) for k, v in losses.items()})


if __name__ == '__main__':
    main()
