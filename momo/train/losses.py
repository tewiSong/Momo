from typing import Dict

import torch
import torch.nn.functional as F


def compute_losses(
    h_motif_2d: torch.Tensor,
    router: Dict[str, torch.Tensor],
    weights: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    # Fail fast: required router fields must exist for pretraining（几何主监督版）
    assert isinstance(router, dict)
    for k in ['logits_code', 'e_hat', 'h_motif_enh', 'z3d_gt', 'zgt_proj', 'z_pred', 'z_proto', 'codebook_codes', 'proto_topk_idx', 'h_motif_local', 'z_template']:
        assert k in router, f'missing router field: {k}'

    device = h_motif_2d.device
    with torch.cuda.amp.autocast(enabled=False):
        logits_code = router['logits_code']
        p_code = F.softmax(logits_code.float(), dim=-1)

        # Base accumulators
        loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        ent = torch.tensor(0.0, dtype=torch.float32, device=device)
        balance = torch.tensor(0.0, dtype=torch.float32, device=device)

        # 1) 几何重建：pred_geom = latent_to_zgt(h_motif_enh) 与 z3d_gt 的 MSE
        pred_geom = router['z_pred'].float()
        z3d_gt = router['z3d_gt'].float()
        recon = F.mse_loss(pred_geom, z3d_gt)
        loss = loss + float(weights.get('recon_weight', 0.0)) * recon

        # 1.1) 原型聚合（模板监督，latent 空间）：直接让 e_hat 贴近 z_template
        z_proto = router['z_proto'].float()  # 仅用于监控，不参与该项损失
        e_hat = router['e_hat'].float()
        z_template_lat = router['z_template'].float().detach()
        proto_recon = F.mse_loss(e_hat, z_template_lat)
        loss = loss + float(weights.get('proto_recon_weight', 0.0)) * proto_recon

        # 2) VQ / Commit（基于模板监督）：
        #    - vq: 使被选原型靠近 z_template（stop-grad on z_template）
        #    - commit: 使 h_motif_2d 靠近前向选中的原型（stop-grad on code）
        vq_w = float(weights.get('vq_weight', 0.0))
        commit_w = float(weights.get('commit_weight', 0.0))
        codes = router['codebook_codes'].float()
        # e_target（模板最近原型 Top‑1）
        if vq_w > 0.0:
            assert 'gt_nn_index' in router, 'vq_weight>0 requires gt_nn_index'
            e_target = codes.index_select(0, router['gt_nn_index'].long())
            z_template = router['z_template'].float().detach()
            vq = torch.mean((e_target - z_template) ** 2)
        else:
            vq = torch.tensor(0.0, dtype=torch.float32, device=device)
        loss = loss + vq_w * vq
        # e_selected（路由 Top‑1 原型，用于 commit，对齐无上下文局部表示）
        if commit_w > 0.0:
            idx_top1 = router['proto_topk_idx'][:, 0].long()
            e_selected = codes.index_select(0, idx_top1)
            commit = torch.mean((router['h_motif_local'].float() - e_selected.detach()) ** 2)
        else:
            commit = torch.tensor(0.0, dtype=torch.float32, device=device)
        loss = loss + commit_w * commit

        # 3) 原型路由监督：KL(p_gt || p_code) 与 CE(k_gt)
        kl = torch.tensor(0.0, dtype=torch.float32, device=device)
        ce = torch.tensor(0.0, dtype=torch.float32, device=device)
        kl_w = float(weights.get('kl_weight', 0.0))
        ce_w = float(weights.get('ce_weight', 0.0))
        if kl_w > 0.0:
            assert 'p_gt' in router, 'kl_weight>0 requires p_gt in router'
            pt = torch.clamp(router['p_gt'].float().detach(), min=1e-12)
            pc = torch.clamp(p_code, min=1e-12)
            kl = torch.sum(pt * (torch.log(pt) - torch.log(pc)), dim=-1).mean()
            loss = loss + kl_w * kl
        if ce_w > 0.0:
            assert 'gt_nn_index' in router, 'ce_weight>0 requires gt_nn_index in router'
            ce = F.cross_entropy(logits_code.float(), router['gt_nn_index'].long())
            loss = loss + ce_w * ce

        # 3.1) Δ 仅补差：让 z_pred 与 z_proto 的差分对齐到 (z3d_gt - z_proto)
        delta_res = torch.tensor(0.0, dtype=torch.float32, device=device)
        dr_w = float(weights.get('delta_res_weight', 0.0))
        if dr_w > 0.0:
            delta_res = F.mse_loss((pred_geom - z_proto), (z3d_gt - z_proto).detach())
            loss = loss + dr_w * delta_res

        # 3.2) Δ 正交约束：抑制 Δ 与 e_hat 共线（弱约束）
        orth = torch.tensor(0.0, dtype=torch.float32, device=device)
        ow = float(weights.get('orth_weight', 0.0))
        if ow > 0.0:
            d = router.get('delta', torch.zeros_like(router['e_hat']).float()).float()
            eh = router['e_hat'].float()
            d_n = F.normalize(d, dim=-1)
            eh_n = F.normalize(eh, dim=-1)
            cos2 = (d_n * eh_n).sum(dim=-1) ** 2
            orth = cos2.mean()
            loss = loss + ow * orth

        # 4) Expert load balancing (batch mean vs uniform)
        exp_balance = torch.tensor(0.0, dtype=torch.float32, device=device)
        eb_w = float(weights.get('exp_balance_weight', 0.0))
        if eb_w > 0.0:
            assert 'p_exp' in router, 'exp_balance_weight>0 requires p_exp in router'
            p_bar = router['p_exp'].float().mean(dim=0)
            M = int(p_bar.numel())
            uniform = torch.full_like(p_bar, 1.0 / float(M))
            exp_balance = torch.sum((p_bar - uniform) ** 2)
            loss = loss + eb_w * exp_balance

        # 5) Δ capacity regularization
        delta_l2 = torch.tensor(0.0, dtype=torch.float32, device=device)
        d_w = float(weights.get('delta_l2_weight', 0.0))
        if d_w > 0.0:
            assert 'delta' in router, 'delta_l2_weight>0 requires delta in router'
            d = router['delta'].float()
            delta_l2 = torch.mean(torch.sum(d * d, dim=-1))
            loss = loss + d_w * delta_l2

        # 6) Prototype-side entropy and batch-balance
        p_code = F.softmax(logits_code.float(), dim=-1)
        ent = torch.sum(-p_code * torch.log(torch.clamp(p_code, min=1e-12)), dim=-1).mean()
        ent_w = float(weights.get('ent_weight', 0.0))
        if ent_w != 0.0:
            loss = loss + ent_w * (-ent)
        p_bar = p_code.mean(dim=0)
        K = int(p_bar.numel())
        uniform = torch.full_like(p_bar, 1.0 / float(K))
        balance = torch.sum((p_bar - uniform) ** 2)
        bal_w = float(weights.get('balance_weight', 0.0))
        loss = loss + bal_w * balance

    return {
        'loss': loss,
        'recon': recon,
        'proto_recon': proto_recon,
        'delta_res': delta_res,
        'orth': orth,
        'vq': vq,
        'commit': commit,
        'kl': kl,
        'ce': ce,
        'ent': ent,
        'balance': balance,
        'exp_balance': exp_balance,
        'delta_l2': delta_l2,
        'geom_align': recon,
    }
