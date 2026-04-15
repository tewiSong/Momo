from typing import Dict

import torch
import torch.nn.functional as F


def compute_losses(
    h_motif_2d: torch.Tensor,
    router: Dict[str, torch.Tensor],
    weights: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    assert isinstance(router, dict)
    for k in [
        'e_hat', 'h_motif_enh', 'edge_target', 'edge_delta_geom',
        'delta', 'delta_ctx', 'z3d_pred', 'z3d_target', 'z_e', 'z_q',
        'z_template_target',
    ]:
        assert k in router, f'missing router field: {k}'

    device = h_motif_2d.device
    with torch.cuda.amp.autocast(enabled=False):
        loss = torch.tensor(0.0, dtype=torch.float32, device=device)

        edge_pred = router['edge_delta_geom'].float()
        edge_tgt = router['edge_target'].float()
        assert edge_pred.shape == edge_tgt.shape
        edge_delta_res = torch.mean(torch.sum((edge_pred - edge_tgt) ** 2, dim=-1))
        loss = loss + float(weights['edge_weight']) * edge_delta_res

        z3d_pred = router['z3d_pred'].float()
        z3d_target = router['z3d_target'].float().detach()
        main_3d = F.mse_loss(z3d_pred, z3d_target)
        loss = loss + float(weights['main_3d_weight']) * main_3d

        z_e = router['z_e'].float()
        z_q = router['z_q'].float()
        z_template_target = router['z_template_target'].float().detach()
        template = F.mse_loss(router['e_hat'].float(), z_template_target)
        loss = loss + float(weights.get('template_weight', 0.0)) * template

        vq = F.mse_loss(z_q, z_e.detach())
        loss = loss + float(weights['vq_weight']) * vq

        commit = F.mse_loss(z_e, z_q.detach())
        loss = loss + float(weights['commit_weight']) * commit

        d = router['delta'].float()
        eh = router['e_hat'].float()
        d_norm = torch.linalg.norm(d, dim=-1, keepdim=True)
        eh_norm = torch.linalg.norm(eh, dim=-1, keepdim=True)
        eps = 1e-12
        d_n = d / (d_norm + eps)
        eh_n = eh / (eh_norm + eps)
        orth = torch.mean(torch.sum(d_n * eh_n, dim=-1) ** 2)
        loss = loss + float(weights['orth_weight']) * orth

        delta_l2 = torch.mean(torch.sum(d * d, dim=-1))
        loss = loss + float(weights['delta_l2_weight']) * delta_l2

        d_ctx = router['delta_ctx'].float()
        ctx_delta_l2 = torch.mean(torch.sum(d_ctx * d_ctx, dim=-1))
        loss = loss + float(weights.get('ctx_delta_l2_weight', 0.0)) * ctx_delta_l2

        dctx_norm = torch.linalg.norm(d_ctx, dim=-1, keepdim=True)
        dctx_n = d_ctx / (dctx_norm + eps)
        ctx_delta_orth = torch.mean(torch.sum(dctx_n * eh_n, dim=-1) ** 2)
        loss = loss + float(weights.get('ctx_delta_orth_weight', 0.0)) * ctx_delta_orth

    return {
        'loss': loss,
        'edge': edge_delta_res,
        'main_3d': main_3d,
        'template': template,
        'vq': vq,
        'commit': commit,
        'orth': orth,
        'delta_l2': delta_l2,
        'ctx_delta_l2': ctx_delta_l2,
        'ctx_delta_orth': ctx_delta_orth,
    }
