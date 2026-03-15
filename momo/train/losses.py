from typing import Dict, Optional

import torch
import torch.nn.functional as F


def compute_losses(
    z_hat: torch.Tensor,
    z_gt: torch.Tensor,
    h_motif_2d: torch.Tensor,
    e_k: torch.Tensor,
    weights: Dict[str, float],
    router: Optional[Dict[str, torch.Tensor]] = None,
    teacher_z: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    # 形状断言
    assert z_hat.shape == z_gt.shape
    assert h_motif_2d.size(0) == z_hat.size(0)
    assert e_k.shape == z_hat.shape

    # 在 FP32 通道计算损失，避免 AMP 下的溢出
    with torch.cuda.amp.autocast(enabled=False):
        z_hat_f = z_hat.float()
        z_gt_f = z_gt.float()
        h2d_f = h_motif_2d.float()
        e_k_f = e_k.float()

        recon = F.mse_loss(z_hat_f, z_gt_f)
        vq = torch.mean(torch.sum((z_gt_f.detach() - e_k_f) ** 2, dim=-1))
        commit = torch.mean(torch.sum((h2d_f - e_k_f.detach()) ** 2, dim=-1))

        loss = (
            float(weights['recon_weight']) * recon
            + float(weights['vq_weight']) * vq
            + float(weights['commit_weight']) * commit
        )

    # 额外的路由相关损失（可选，由 cfg 权重控制，不硬编码）
        kl = torch.tensor(0.0, dtype=torch.float32, device=z_hat.device)
        ce = torch.tensor(0.0, dtype=torch.float32, device=z_hat.device)
        ent = torch.tensor(0.0, dtype=torch.float32, device=z_hat.device)
        if router is not None:
            # 将监督与正则统一到“真正用于选码”的分布 y_soft
            y_soft = router.get('y_soft')
            p_t = router.get('p_t')  # 教师/几何分布（teacher_z 或 z_gt 生成）
            logits = router.get('logits')
            teacher_nn_index = router.get('teacher_nn_index')
            eps = 1e-8
            if y_soft is not None and p_t is not None:
                ys = torch.clamp(y_soft.float(), min=eps)
                pt = torch.clamp(p_t.float(), min=eps)
                # 双向 KL：匹配 y_soft 与教师分布
                kl1 = torch.sum(pt * (torch.log(pt) - torch.log(ys)), dim=-1).mean()
                kl2 = torch.sum(ys * (torch.log(ys) - torch.log(pt)), dim=-1).mean()
                kl = kl1 + kl2
                # 直接在 y_soft 上做熵正则，鼓励使用更多 code
                ent = -torch.sum(ys * torch.log(ys), dim=-1).mean()
            # 可选：还保留一个与教师最近邻的 CE 监督（默认权重为 0）
            if logits is not None and teacher_nn_index is not None:
                ce = F.cross_entropy(logits.float(), teacher_nn_index.long())

            kl_w = float(weights.get('kl_weight', 0.0))
            ce_w = float(weights.get('ce_weight', 0.0))
            ent_w = float(weights.get('ent_weight', 0.0))
            loss = loss + kl_w * kl + ce_w * ce + ent_w * ent

        # Teacher 相关损失（显式权重控制）
        vq_teacher = torch.tensor(0.0, dtype=torch.float32, device=z_hat.device)
        recon_teacher = torch.tensor(0.0, dtype=torch.float32, device=z_hat.device)
        teacher_align = torch.tensor(0.0, dtype=torch.float32, device=z_hat.device)
        if teacher_z is not None:
            tz = teacher_z.float()
            # 1) 对齐：显式将 teacher 映射到与 z_gt 同一语义/尺度空间
            #    该项允许反传到 teacher（或仅到投影层，取决于参数 requires_grad）。
            teacher_align = F.mse_loss(tz, z_gt_f)
            ta_w = float(weights.get('teacher_align_weight', 0.0))
            loss = loss + ta_w * teacher_align

            # 2) 可选蒸馏/引导：保持历史字段，但默认配置将权重置为 0。
            #    - vq_teacher 仅拉动 codebook，避免 teacher 不稳时误导学生。
            #    - recon_teacher 采用 detach，避免该项更新 teacher。
            vq_teacher = torch.mean(torch.sum((tz.detach() - e_k_f) ** 2, dim=-1))
            recon_teacher = F.mse_loss(z_hat_f, tz.detach())
            vqt_w = float(weights.get('vq_teacher_weight', 0.0))
            rct_w = float(weights.get('recon_teacher_weight', 0.0))
            loss = loss + vqt_w * vq_teacher + rct_w * recon_teacher

    return {
        'loss': loss,
        'recon': recon,
        'vq': vq,
        'commit': commit,
        'kl': kl,
        'ce': ce,
        'ent': ent,
        'vq_teacher': vq_teacher,
        'recon_teacher': recon_teacher,
        'teacher_align': teacher_align,
    }
