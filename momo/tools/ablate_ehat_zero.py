#!/usr/bin/env python
"""
Run e_hat=0 ablation on a single batch and report:
- ||Delta(normal) - Delta(e_hat=0)||
- ||h_motif_enh(normal) - h_motif_enh(e_hat=0)||
- CE/MMM/Graph losses (normal vs ablated) and their differences

No fallbacks or try/except: assertions will fail fast if configs/data unsupported.
"""
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from momo.utils.config import load_yaml
from momo.data.pcqm4mv2 import PCQM4Mv2MotifDataset, _infer_num_motifs_per_graph
from momo.models.gin_motif_vqmoe import MotifVQMoE


def _graph_readout_sum(feat: torch.Tensor, motif_id: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    g_counts = _infer_num_motifs_per_graph(motif_id, batch)
    motif_graph_ids = torch.repeat_interleave(torch.arange(g_counts.numel(), device=g_counts.device), g_counts)
    outg = torch.zeros((g_counts.numel(), feat.size(1)), dtype=feat.dtype, device=feat.device)
    outg = outg.index_add(0, motif_graph_ids, feat)
    return outg


def _nt_xent(a: torch.Tensor, p: torch.Tensor, t: float) -> torch.Tensor:
    a_n = F.normalize(a, dim=-1)
    p_n = F.normalize(p, dim=-1)
    logits = a_n @ p_n.t() / t
    targets = torch.arange(a.size(0), device=a.device)
    return F.cross_entropy(logits, targets)


def _compute_delta_from_experts(model: MotifVQMoE,
                                h2d: torch.Tensor,
                                e_hat: torch.Tensor,
                                p_exp: torch.Tensor,
                                exp_idx: torch.Tensor) -> torch.Tensor:
    # Mirror forward() expert combine: top-k indices in exp_idx, weights from p_exp normalized over top-k
    assert hasattr(model, 'experts') and isinstance(model.experts, torch.nn.ModuleList)
    k = int(exp_idx.size(1))
    exp_vals = torch.gather(p_exp, dim=1, index=exp_idx)
    exp_vals_norm = exp_vals / exp_vals.sum(dim=-1, keepdim=True)
    expert_inputs = torch.cat([h2d, e_hat], dim=-1)
    deltas = []
    for j in range(k):
        ej_idx = exp_idx[:, j]
        parts = []
        for i in range(len(model.experts)):
            sel = (ej_idx == i)
            if sel.any():
                out_i = model.experts[i](expert_inputs[sel])
                parts.append((sel, out_i))
        delta_j = torch.zeros_like(h2d)
        for sel, val in parts:
            delta_j[sel] = val
        deltas.append(delta_j)
    delta = torch.zeros_like(h2d)
    for j in range(k):
        wj = exp_vals_norm[:, j].view(-1, 1)
        delta = delta + wj * deltas[j]
    return delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', type=str, required=True)
    ap.add_argument('--batch-size', type=int, default=8)
    args = ap.parse_args()

    cfg = load_yaml(args.cfg)
    device = 'cuda' if (cfg['misc'].get('device', 'cuda') == 'cuda' and torch.cuda.is_available()) else 'cpu'

    # Require teacher for CE/MMM supervision
    assert bool(cfg.get('teacher', {}).get('enabled', True)) is True, 'teacher.enabled must be true for this ablation'

    ds = PCQM4Mv2MotifDataset(
        preprocessed_path=cfg['dataset']['preprocessed_path'],
        z3d_dim=cfg['model']['z3d_dim'],
        max_atomic_num=cfg['dataset']['max_atomic_num'],
        require_pos=True,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=cfg['dataset']['num_workers'])

    model = MotifVQMoE(cfg).to(device)
    model.eval()

    batch = next(iter(dl)).to(device)

    # Base forward (no mask)
    h2d, h_enh, info = model(batch)
    assert 'delta' in info and 'e_hat' in info and 'p_exp' in info and 'exp_topk_idx' in info
    delta_base = info['delta']
    e_hat = info['e_hat']
    p_exp = info['p_exp']
    exp_idx = info['exp_topk_idx']
    delta_zero_raw = _compute_delta_from_experts(model, h2d, torch.zeros_like(e_hat), p_exp, exp_idx)
    h_enh_zero = torch.zeros_like(h2d) + (model.delta_scale * delta_zero_raw)

    # Norm differences
    d_delta = torch.linalg.norm(delta_base - (model.delta_scale * delta_zero_raw), ord=2, dim=1).mean()
    d_henh = torch.linalg.norm(h_enh - h_enh_zero, ord=2, dim=1).mean()

    # CE (prototype classification) normal vs ablated (identical by construction)
    ce_normal = F.cross_entropy(info['logits_code'].float(), info['teacher_nn_index'].long())
    ce_zero = ce_normal

    # MMM (masked motif modeling)
    mmm_ratio = float(cfg.get('mmm', {}).get('mask_ratio', 0.15))
    n = int(h2d.size(0))
    msk = (torch.rand(n, device=device) < mmm_ratio)
    assert msk.any(), 'mask produced no true entries'
    _, _, info_m = model(batch, motif_mask=msk)
    k_target = info['teacher_nn_index']
    mmm_normal = F.cross_entropy(info_m['logits_code'][msk].float(), k_target[msk].long())
    mmm_zero = mmm_normal  # prototype logits do not depend on e_hat

    # Graph contrastive (TMCL-like) normal vs ablated
    graph = dict(cfg.get('graph', {}))
    graph_ratio_l = float(graph.get('mask_ratio_light', mmm_ratio))
    graph_ratio_h = float(graph.get('mask_ratio_heavy', mmm_ratio))
    graph_temp = float(graph.get('temperature', 0.2))
    m1 = (torch.rand(n, device=device) < graph_ratio_l)
    m2 = (torch.rand(n, device=device) < graph_ratio_h)
    _, _, info_m1 = model(batch, motif_mask=m1)
    _, _, info_m2 = model(batch, motif_mask=m2)

    # Normal graph loss
    g0 = _graph_readout_sum(h_enh, batch.motif_id, batch.batch)
    g1 = _graph_readout_sum(info_m1['h_motif_enh'], batch.motif_id, batch.batch)
    g2 = _graph_readout_sum(info_m2['h_motif_enh'], batch.motif_id, batch.batch)
    graph_normal = 0.5 * (_nt_xent(g0, g1, graph_temp) + _nt_xent(g0, g2, graph_temp))

    # Ablated graph loss: recompute h_enh_zero for base and masked views
    g0z = _graph_readout_sum(h_enh_zero, batch.motif_id, batch.batch)
    # We must re-run masked forwards to obtain masked h2d and expert routing
    h2d_m1, h_enh_m1, info_m1_again = model(batch, motif_mask=m1)
    h2d_m2, h_enh_m2, info_m2_again = model(batch, motif_mask=m2)
    delta_m1_zero_raw = _compute_delta_from_experts(model, h2d_m1, torch.zeros_like(info_m1_again['e_hat']), info_m1_again['p_exp'], info_m1_again['exp_topk_idx'])
    delta_m2_zero_raw = _compute_delta_from_experts(model, h2d_m2, torch.zeros_like(info_m2_again['e_hat']), info_m2_again['p_exp'], info_m2_again['exp_topk_idx'])
    h_enh_m1_zero = torch.zeros_like(h2d_m1) + (model.delta_scale * delta_m1_zero_raw)
    h_enh_m2_zero = torch.zeros_like(h2d_m2) + (model.delta_scale * delta_m2_zero_raw)
    g1z = _graph_readout_sum(h_enh_m1_zero, batch.motif_id, batch.batch)
    g2z = _graph_readout_sum(h_enh_m2_zero, batch.motif_id, batch.batch)
    graph_zero = 0.5 * (_nt_xent(g0z, g1z, graph_temp) + _nt_xent(g0z, g2z, graph_temp))

    print('Ablation Report (means)')
    print(f"||Delta(normal)-Delta(e_hat=0)||: {float(d_delta):.6f}")
    print(f"||h_enh(normal)-h_enh(e_hat=0)||: {float(d_henh):.6f}")
    print(f"CE normal: {float(ce_normal):.6f}  CE ablated: {float(ce_zero):.6f}  diff: {float((ce_zero - ce_normal)):.6f}")
    print(f"MMM normal: {float(mmm_normal):.6f} MMM ablated: {float(mmm_zero):.6f} diff: {float((mmm_zero - mmm_normal)):.6f}")
    print(f"Graph normal: {float(graph_normal):.6f} Graph ablated: {float(graph_zero):.6f} diff: {float((graph_zero - graph_normal)):.6f}")


if __name__ == '__main__':
    main()
