# momo/train/pretrain.py
import os
import argparse
from typing import Dict, Optional
from datetime import datetime
import socket

import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from momo.utils.config import load_yaml
from momo.data.pcqm4mv2 import PCQM4Mv2MotifDataset
from momo.models.gin_motif_vqmoe import MotifVQMoE
from momo.train.losses import compute_losses
import torch.nn.functional as F


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _reservoir_sample_z3d(ds: PCQM4Mv2MotifDataset, sample_size: int, z3d_dim: int, seed: int) -> torch.Tensor:
    """Uniformly sample up to `sample_size` motif 3D targets from the dataset using reservoir sampling.

    Returns a CPU float32 tensor of shape [N, z3d_dim].
    """
    import random

    random.seed(seed)
    buf = []
    seen = 0
    for rec in ds.data:  # type: ignore[attr-defined]
        feats = rec['motif_features']  # List[List[float]]
        for v in feats:
            seen += 1
            if len(buf) < sample_size:
                buf.append(v)
            else:
                j = random.randrange(seen)
                if j < sample_size:
                    buf[j] = v
    if len(buf) == 0:
        raise RuntimeError('No motif_features found in dataset for k-means init')
    X = torch.tensor(buf, dtype=torch.float32)
    assert X.dim() == 2 and X.size(1) == z3d_dim, f"Unexpected z3d_dim: got {X.size(1)}, expect {z3d_dim}"
    # 标准化到与训练一致的空间
    X = (X - ds.z3d_mean.cpu()) / ds.z3d_std.cpu()
    return X


def _kmeans_torch(X: torch.Tensor, K: int, max_iters: int = 30, tol: float = 1e-4, seed: int = 42) -> torch.Tensor:
    """Simple k-means on CPU with mini-batch assignment to avoid O(N*K) memory explosion.

    X: [N, D] float32 CPU tensor
    Returns centroids [K, D]
    """
    assert X.device.type == 'cpu'
    N, D = X.size(0), X.size(1)
    g = torch.Generator(device='cpu').manual_seed(seed)
    if N < K:
        # If points fewer than clusters, pad with repeated samples
        idx = torch.arange(N)
        reps = (K + N - 1) // N
        idx = idx.repeat(reps)[:K]
    else:
        idx = torch.randperm(N, generator=g)[:K]
    C = X[idx].clone()  # [K, D]

    prev_C = C.clone()
    assign = torch.empty(N, dtype=torch.long)
    batch = 4096  # assignment mini-batch size
    for it in range(max_iters):
        # Assign step
        for s in range(0, N, batch):
            e = min(s + batch, N)
            xb = X[s:e]  # [B, D]
            # distances to centroids: (xb^2 + C^2 - 2 xb C^T)
            # compute via (xb @ C^T)
            x2 = (xb * xb).sum(dim=1, keepdim=True)  # [B,1]
            c2 = (C * C).sum(dim=1).view(1, -1)      # [1,K]
            d2 = x2 + c2 - 2.0 * xb.matmul(C.t())    # [B,K]
            assign[s:e] = torch.argmin(d2, dim=1)

        # Update step
        C.zero_()
        counts = torch.zeros(K, dtype=torch.long)
        for k in range(K):
            mask = (assign == k)
            cnt = int(mask.sum().item())
            if cnt > 0:
                C[k] = X[mask].mean(dim=0)
                counts[k] = cnt
        # Handle empty clusters by re-seeding to random points
        empty = (counts == 0).nonzero(as_tuple=False).view(-1)
        if empty.numel() > 0:
            ridx = torch.randperm(N, generator=g)[:empty.numel()]
            C[empty] = X[ridx]

        # Check convergence via centroid shift
        shift = torch.norm(C - prev_C, p=2) / (torch.norm(prev_C, p=2) + 1e-12)
        if shift.item() < tol:
            break
        prev_C.copy_(C)
    return C


def eval_epoch(model: torch.nn.Module,
               dl: DataLoader,
               device: str,
               loss_cfg: Dict[str, float]) -> Dict[str, float]:
    model.eval()
    tot = 0
    sums = {k: 0.0 for k in ['loss', 'recon', 'proto_recon', 'delta_res', 'orth', 'vq', 'commit', 'kl', 'ce', 'ent', 'balance', 'geom_align', 'exp_balance', 'delta_l2']}
    with torch.no_grad():
        for batch in dl:
            batch = batch.to(device)
            h_motif_2d, h_motif_enh, router_info = model(batch)
            out = compute_losses(h_motif_2d=h_motif_2d, router=router_info, weights=loss_cfg)
            bs = h_motif_2d.size(0)
            tot += bs
            for k in sums.keys():
                sums[k] += float(out[k].detach().cpu()) * bs
    return {k: (v / max(1, tot)) for k, v in sums.items()}


def _save_ckpt(path: str,
               model: torch.nn.Module,
               optim: torch.optim.Optimizer,
               scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
               epoch: int,
               global_step: int,
               best_val: float,
               scaler: Optional[torch.cuda.amp.GradScaler]) -> None:
    state = {
        'model': model.state_dict(),
        'optimizer': optim.state_dict(),
        'epoch': epoch,
        'step': global_step,
        'best_val': best_val,
    }
    if scheduler is not None:
        state['scheduler'] = scheduler.state_dict()
    if scaler is not None:
        state['scaler'] = scaler.state_dict()
    torch.save(state, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', type=str, required=True)
    p.add_argument('--run_name', type=str, default='pretrain')
    p.add_argument('--resume', type=str, default='')
    args = p.parse_args()

    cfg = load_yaml(args.cfg)
    wanted = cfg['misc'].get('device', 'cuda')
    device = 'cuda' if (wanted == 'cuda' and torch.cuda.is_available()) else ('cpu' if wanted == 'cuda' else wanted)
    setup_seed(int(cfg['misc'].get('seed', 42)))

    # Dataset and split
    ds = PCQM4Mv2MotifDataset(
        preprocessed_path=cfg['dataset']['preprocessed_path'],
        z3d_dim=cfg['model']['geom_dim'],
        max_atomic_num=cfg['dataset']['max_atomic_num'],
        require_pos=bool(cfg.get('teacher', {}).get('enabled', False)),
    )
    print(f"Loaded dataset from {cfg['dataset']['preprocessed_path']} with {len(ds)} molecules")
    assert int(ds.z3d_dim) == int(cfg['model']['geom_dim']), (
        f"geom_dim mismatch: dataset={int(ds.z3d_dim)} vs cfg.model.geom_dim={int(cfg['model']['geom_dim'])}"
    )

    val_ratio = float(cfg['dataset'].get('val_ratio', 0.05))
    n_total = len(ds)
    n_val = max(1, int(round(n_total * val_ratio)))
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(int(cfg['misc'].get('seed', 42)))
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=gen)

    dl_train = DataLoader(
        train_ds,
        batch_size=cfg['dataset']['batch_size'],
        shuffle=cfg['dataset']['shuffle'],
        num_workers=cfg['dataset']['num_workers'],
    )
    dl_val = DataLoader(
        val_ds,
        batch_size=cfg['dataset']['batch_size'],
        shuffle=False,
        num_workers=cfg['dataset']['num_workers'],
    )

    # Model & optim（codebook 初始化参数仅记录，实际在模型构建后执行）
    init_cfg = cfg['model'].get('codebook_init', {})
    init_name = str(init_cfg.get('name', 'random')).lower()
    kmeans_centroids = None

    model = MotifVQMoE(cfg).to(device)
    # 冻结/解冻 teacher 参数：从根源修复需先把 teacher 约束到 z_gt，同期避免其不稳定牵动主干
    tcfg = cfg.get('teacher', {})
    trainable = bool(tcfg.get('trainable', False))
    proj_trainable = bool(tcfg.get('proj_trainable', True))
    if hasattr(model, 'teacher') and model.teacher is not None:
        for p in model.teacher.parameters():
            p.requires_grad = trainable
    if hasattr(model, 'teacher_proj') and model.teacher_proj is not None:
        for p in model.teacher_proj.parameters():
            p.requires_grad = proj_trainable
    # Codebook 初始化（基于几何投影到 latent 的 k-means）
    if init_name == 'kmeans':
        sample_size = int(init_cfg.get('sample_size', 200000))
        max_iters = int(init_cfg.get('max_iters', 30))
        tol = float(init_cfg.get('tol', 1e-4))
        seed = int(cfg['misc'].get('seed', 42))
        print(f"[Init] Collecting up to {sample_size} z3d_gt samples for k-means (projected to latent)...")
        X = _reservoir_sample_z3d(ds, sample_size, int(ds.z3d_dim), seed)  # [N, geom_dim]
        with torch.no_grad():
            Z = model.zgt_to_latent(X.to(device)).detach().cpu().to(torch.float32)
        K = int(cfg['model']['codebook_size'])
        N_eff = Z.size(0)
        print(f"[Init] Running k-means on {N_eff} projected samples into {K} clusters (iters={max_iters}, tol={tol})")
        kmeans_centroids = _kmeans_torch(Z, K=K, max_iters=max_iters, tol=tol, seed=seed)
        with torch.no_grad():
            assert kmeans_centroids.shape == model.codebook.codes.data.shape
            model.codebook.codes.data.copy_(kmeans_centroids.to(model.codebook.codes.data.dtype))
        # Optional inertia
        with torch.no_grad():
            batch = 8192
            sse = 0.0
            for i in range(0, N_eff, batch):
                zb = Z[i:i+batch]
                z2 = (zb * zb).sum(dim=1, keepdim=True)
                c2 = (kmeans_centroids * kmeans_centroids).sum(dim=1).view(1, -1)
                d2 = z2 + c2 - 2.0 * zb.matmul(kmeans_centroids.t())
                sse += float(d2.min(dim=1).values.sum().item())
        print(f"[Init] k-means inertia (projected): {sse:.4f}")
        print("[Init] Codebook initialized from k-means centroids (projected).")
    optim = torch.optim.Adam(model.parameters(), lr=float(cfg['optim']['lr']))

    # Scheduler
    sch_cfg = cfg.get('scheduler', {})
    sch_name = str(sch_cfg.get('name', 'none')).lower()
    scheduler = None
    if sch_name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim,
            step_size=int(sch_cfg.get('step_size', 10)),
            gamma=float(sch_cfg.get('gamma', 0.5)),
        )
    elif sch_name == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode='min', factor=float(sch_cfg.get('gamma', 0.5)), patience=int(sch_cfg.get('patience', 5))
        )
    elif sch_name == 'cosine':
        # Cosine 退火（按 epoch 步进），更平滑稳定
        t_max = int(sch_cfg.get('t_max_epochs', cfg['misc'].get('epochs', 50)))
        eta_min = float(sch_cfg.get('eta_min', 0.0))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=t_max, eta_min=eta_min)

    # TensorBoard: 独立子目录避免多次运行混写
    log_dir = cfg['misc'].get('log_dir', 'runs')
    os.makedirs(log_dir, exist_ok=True)
    run_suffix = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{socket.gethostname()}"
    run_id = f"{args.run_name}-{run_suffix}"
    writer = SummaryWriter(log_dir=os.path.join(log_dir, run_id))

    epochs = int(cfg['misc'].get('epochs', 50))
    log_every = int(cfg['misc'].get('log_every_steps', 50))
    eval_every = int(cfg['misc'].get('eval_every_epochs', 1))
    eval_every_steps = int(cfg['misc'].get('eval_every_steps', 0))  # 0 表示按 epoch 评估
    # 两阶段训练：前期冻结 Δ（严格要求在配置中提供步数）
    delta_freeze_steps = int(cfg['model']['delta_freeze_steps'])

    # 基本权重与简化调度（仅保留 commit 升温与熵/均衡后期退火）
    loss_base = dict(cfg['loss'])
    # commit 权重线性升温步数（避免早期强绑定码本）
    commit_ramp_steps = int(cfg.get('loss', {}).get('commit_ramp_steps', 50000))

    def _apply_weight_schedule(base_w: float, sched: Dict[str, float], step: int) -> float:
        """线性调度辅助：从 base_w 过渡到 sched['end_weight']，
        在 sched['start_step'] 开始、历时 sched['duration'] 步完成。若未提供则返回 base_w。
        """
        if not isinstance(sched, dict):
            return float(base_w)
        end_w = float(sched.get('end_weight', base_w))
        start = int(sched.get('start_step', -1))
        dur = int(sched.get('duration', 0))
        if start < 0 or dur <= 0:
            return float(base_w)
        if step <= start:
            return float(base_w)
        p = min(1.0, max(0.0, float(step - start) / float(dur)))
        return float(base_w) + (end_w - float(base_w)) * p

    def current_weights(step: int):
        w = dict(loss_base)
        # commit 权重升温
        if commit_ramp_steps > 0:
            p_commit = min(1.0, float(step) / float(commit_ramp_steps))
            w['commit_weight'] = float(loss_base.get('commit_weight', 0.0)) * p_commit
        # 熵/均衡后期退火
        ent_base = float(loss_base.get('ent_weight', 0.0))
        bal_base = float(loss_base.get('balance_weight', 0.0))
        w['ent_weight'] = _apply_weight_schedule(ent_base, cfg['loss'].get('ent_schedule', {}), step)
        w['balance_weight'] = _apply_weight_schedule(bal_base, cfg['loss'].get('balance_schedule', {}), step)
        return w

    # 路由调度参数（必须配置，不做静默回退）
    rcfg = cfg.get('router', None)
    assert rcfg is not None, 'router 配置缺失'
    tau_r_start = float(rcfg['gumbel_tau_start'])
    tau_r_end = float(rcfg['gumbel_tau_end'])
    beta_start = float(rcfg['beta_start'])
    beta_end = float(rcfg['beta_end'])
    warmup_steps = int(rcfg['warmup_steps'])
    topk_warmup_steps = int(rcfg.get('topk_warmup_steps', 0))
    # 可选：教师温度退火（高→低），以及晚期将 β 提升到更大值（保留，但不强制要求 teacher）
    tt_start = float(rcfg.get('teacher_temperature_start', rcfg.get('teacher_temperature', 1.0)))
    tt_end = float(rcfg.get('teacher_temperature_end', tt_start))
    tt_decay_start = int(rcfg.get('teacher_temp_decay_start', -1))
    tt_decay_steps = int(rcfg.get('teacher_temp_decay_steps', 0))
    beta_boost = rcfg.get('beta_boost', None)
    beta_boost_start = int(beta_boost.get('start_step', -1)) if isinstance(beta_boost, dict) else -1
    beta_boost_dur = int(beta_boost.get('duration', 0)) if isinstance(beta_boost, dict) else 0
    beta_boost_end = float(beta_boost.get('end_beta', beta_end)) if isinstance(beta_boost, dict) else beta_end

    # 可选：训练后期提升 tau_r 的二阶段调度，提升码本使用度
    boost_cfg = rcfg.get('tau_r_boost', None)
    boost_start = int(boost_cfg.get('start_step', -1)) if isinstance(boost_cfg, dict) else -1
    boost_dur = int(boost_cfg.get('duration', 0)) if isinstance(boost_cfg, dict) else 0
    boost_end_tau = float(boost_cfg.get('end_tau', tau_r_end)) if isinstance(boost_cfg, dict) else tau_r_end

    # Prototype Top‑K 调度（仅保留 ST 硬/软混合，前向只用被选 codes）
    proto_sched = rcfg.get('topk_schedule', None)
    if isinstance(proto_sched, dict):
        stage1_until = int(proto_sched.get('stage1_until', -1))
        stage2_until = int(proto_sched.get('stage2_until', -1))
        stage2_k = int(proto_sched.get('stage2_k', rcfg.get('topk', 1)))
        final_k = int(proto_sched.get('final_k', 1))
        assert stage2_k >= 1 and final_k >= 1, 'proto top-k must be >=1'
    else:
        stage1_until = -1
        stage2_until = -1
        stage2_k = int(rcfg.get('topk', 1))
        final_k = stage2_k

    # 熵目标闭环控制参数（可选）
    ent_target = float(rcfg.get('entropy_target', 0.0))  # 0 表示关闭
    ent_ctrl_lr = float(rcfg.get('entropy_ctrl_lr', 0.0))
    tau_r_min = float(rcfg.get('tau_r_min', 0.4))
    tau_r_max = float(rcfg.get('tau_r_max', 1.5))
    tau_r_state = None  # 维护一个持久化的 tau_r 状态，避免每步被基线调度完全覆盖

    # AMP
    use_amp = bool(cfg['misc'].get('use_amp', True)) and device == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    grad_clip_norm = float(cfg['misc'].get('grad_clip_norm', 1.0))

    # Checkpoint directory（按运行隔离）
    save_root = cfg['misc'].get('save_dir', 'checkpoints')
    save_root_real = os.path.realpath(save_root)
    os.makedirs(save_root_real, exist_ok=True)
    run_dir = os.path.join(save_root_real, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Resume
    resume_path = args.resume or str(cfg['misc'].get('resume') or '')
    start_epoch = 1
    global_step = 0
    # MMM/TMCL 超参（统一从 cfg 读取）
    mmm_cfg = dict(cfg.get('mmm', {}))
    mmm_ratio = float(mmm_cfg.get('mask_ratio', 0.15))
    graph_cfg = dict(cfg.get('graph', {}))
    graph_ratio_l = float(graph_cfg.get('mask_ratio_light', mmm_ratio))
    graph_ratio_h = float(graph_cfg.get('mask_ratio_heavy', mmm_ratio))
    graph_temp = float(graph_cfg.get('temperature', 0.2))
    graph_loss_type = str(graph_cfg.get('loss', 'tmcl')).lower()
    graph_margin = float(graph_cfg.get('margin', 0.2))
    best_val = float('inf')
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optim.load_state_dict(ckpt['optimizer'])
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        global_step = int(ckpt.get('step', 0))
        best_val = float(ckpt.get('best_val', float('inf')))
        if scheduler is not None and 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        if scaler is not None and 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])

    save_every = int(cfg['misc'].get('save_every_epochs', 5))
    save_best_only = bool(cfg['misc'].get('save_best_only', False))
    early_patience = int(cfg['misc'].get('early_stop_patience', 0))
    min_delta = float(cfg['misc'].get('min_delta', 0.0))

    epochs_no_improve = 0
    # 估计总训练步数用于调度（基于当前划分，不跨 run 复用）
    est_total_steps = epochs * max(1, len(dl_train))

    for epoch in range(start_epoch, epochs + 1):
        print(f"Epoch {epoch}/{epochs} - training...")
        model.train()
        # 保险丝持久化状态（跨 step 保持）
        safeguard_until_step = -1
        sg_bad_count = 0
        sg_good_count = 0
        for it, batch in enumerate(dl_train, start=1):
            batch = batch.to(device)
            # 按步应用损失权重与调度（熵/均衡/commit 等）
            weights_cur = current_weights(global_step)
            # Prototype Top‑K 调度：前期使用更宽的 top‑k，后期收敛为 1
            if isinstance(proto_sched, dict):
                if stage2_until > 0 and global_step < stage2_until:
                    k_cur = stage2_k
                else:
                    k_cur = final_k
                if hasattr(model, 'set_proto_topk'):
                    model.set_proto_topk(int(k_cur))
            # 教师温度退火（若启用），按步更新到模型
            if hasattr(model, 'set_teacher_temperature') and getattr(model, 'teacher_enabled', False):
                if tt_decay_start >= 0 and tt_decay_steps > 0:
                    if global_step < tt_decay_start:
                        tt = tt_start
                    else:
                        p = min(1.0, max(0.0, float(global_step - tt_decay_start) / float(max(1, tt_decay_steps))))
                        tt = tt_start + (tt_end - tt_start) * p
                else:
                    tt = float(rcfg.get('teacher_temperature', tt_start))
                model.set_teacher_temperature(float(tt))
            with torch.cuda.amp.autocast(enabled=use_amp):
                freeze_now = (global_step < delta_freeze_steps)
                h_motif_2d, h_enh, router_info = model(batch, freeze_delta=freeze_now)
                losses = compute_losses(h_motif_2d=h_motif_2d, router=router_info, weights=weights_cur)
                loss_total = losses['loss']

                # 1) Masked Motif Modeling (MMM) —— 目标使用 gt_nn_index（几何最近原型）
                mmm_w = float(cfg.get('loss', {}).get('mmm_weight', 0.0))
                if mmm_w > 0.0:
                    num_motifs = int(batch.motif_target.size(0))
                    prob = torch.full((num_motifs,), mmm_ratio, device=batch.motif_target.device)
                    msk = torch.bernoulli(prob).bool()
                    if msk.any():
                        _, _, info_m = model(batch, motif_mask=msk, freeze_delta=freeze_now)
                        assert 'gt_nn_index' in router_info, 'router_info.gt_nn_index missing for MMM target'
                        k_target = router_info['gt_nn_index']
                        ce_mmm = F.cross_entropy(info_m['logits_code'][msk].float(), k_target[msk].long())
                        loss_total = loss_total + mmm_w * ce_mmm
                        losses['mmm'] = ce_mmm

                # 2) Graph-level masked contrastive (TMCL 风格)
                g_w = float(cfg.get('loss', {}).get('graph_weight', 0.0))
                if g_w > 0.0:
                    num_motifs = int(batch.motif_target.size(0))
                    prob_l = torch.full((num_motifs,), graph_ratio_l, device=batch.motif_target.device)
                    prob_h = torch.full((num_motifs,), graph_ratio_h, device=batch.motif_target.device)
                    m1 = torch.bernoulli(prob_l).bool()
                    m2 = torch.bernoulli(prob_h).bool()
                    # 两个掩码视图
                    _, _, info_m1 = model(batch, motif_mask=m1, freeze_delta=freeze_now)
                    _, _, info_m2 = model(batch, motif_mask=m2, freeze_delta=freeze_now)
                    # motif → graph READOUT（sum/mean）
                    from momo.data.pcqm4mv2 import _infer_num_motifs_per_graph
                    g_counts = _infer_num_motifs_per_graph(batch.motif_id, batch.batch)
                    motif_graph_ids = torch.repeat_interleave(torch.arange(g_counts.numel(), device=g_counts.device), g_counts)
                    def _graph_readout(feat: torch.Tensor) -> torch.Tensor:
                        outg = torch.zeros((g_counts.numel(), feat.size(1)), dtype=feat.dtype, device=feat.device)
                        outg = outg.index_add(0, motif_graph_ids, feat)
                        if str(cfg['model'].get('readout','sum')).lower() == 'mean':
                            denom = g_counts.clamp(min=1).view(-1, 1).to(outg.dtype)
                            outg = outg / denom
                        return outg
                    h_enh = router_info['h_motif_enh']
                    g0 = _graph_readout(h_enh)
                    g1 = _graph_readout(info_m1['h_motif_enh'])
                    g2 = _graph_readout(info_m2['h_motif_enh'])
                    # 对比损失（采用 NT-Xent 近似 TMCL）
                    def _nt_xent(a: torch.Tensor, p: torch.Tensor, t: float) -> torch.Tensor:
                        a_n = F.normalize(a, dim=-1)
                        p_n = F.normalize(p, dim=-1)
                        logits = a_n @ p_n.t() / max(t, 1e-8)
                        targets = torch.arange(a.size(0), device=a.device)
                        return F.cross_entropy(logits, targets)
                    loss_graph = 0.5 * (_nt_xent(g0, g1, graph_temp) + _nt_xent(g0, g2, graph_temp))
                    loss_total = loss_total + g_w * loss_graph
                    losses['graph'] = loss_graph

            # 不做数值兜底，保持快速失败

            optim.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss_total).backward()
                # 先反缩放再裁剪
                if grad_clip_norm and grad_clip_norm > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optim)
                scaler.update()
            else:
                loss_total.backward()
                if grad_clip_norm and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optim.step()

            # TensorBoard logging (train)
            if global_step % log_every == 0:
                # 记录实际反传总损与主链路基础损
                writer.add_scalar('train/loss', float(loss_total.detach().cpu()), global_step)
                writer.add_scalar('train/loss_base', float(losses['loss'].detach().cpu()), global_step)
                writer.add_scalar('train/recon', float(losses['recon'].detach().cpu()), global_step)
                writer.add_scalar('train/proto_recon', float(losses['proto_recon'].detach().cpu()), global_step)
                writer.add_scalar('train/delta_res', float(losses['delta_res'].detach().cpu()), global_step)
                writer.add_scalar('train/orth', float(losses['orth'].detach().cpu()), global_step)
                writer.add_scalar('train/vq', float(losses['vq'].detach().cpu()), global_step)
                writer.add_scalar('train/commit', float(losses['commit'].detach().cpu()), global_step)
                writer.add_scalar('train/lr', optim.param_groups[0]['lr'], global_step)
                if 'mmm' in losses:
                    writer.add_scalar('train/mmm', float(losses['mmm'].detach().cpu()), global_step)
                if 'graph' in losses:
                    writer.add_scalar('train/graph', float(losses['graph'].detach().cpu()), global_step)
                writer.add_scalar('train/kl', float(losses['kl'].detach().cpu()), global_step)
                writer.add_scalar('train/ce', float(losses['ce'].detach().cpu()), global_step)
                writer.add_scalar('train/ent', float(losses['ent'].detach().cpu()), global_step)
                writer.add_scalar('train/balance', float(losses['balance'].detach().cpu()), global_step)
                writer.add_scalar('train/geom_align', float(losses['geom_align'].detach().cpu()), global_step)
                # 原型 Top‑1 命中率与使用度（几何目标）
                assert 'gt_nn_index' in router_info, 'router_info.gt_nn_index missing'
                sel = torch.argmax(router_info['p_code'].detach(), dim=-1)
                nn_idx = router_info['gt_nn_index'].detach()
                hit = (sel == nn_idx).float().mean().item()
                writer.add_scalar('train/router/top1_nn_hit', hit, global_step)
                used = torch.unique(sel).numel() / float(model.codebook_size)
                writer.add_scalar('train/router/usage_unique_ratio', used, global_step)
                # 有效代码数
                y_soft = router_info['y_soft'].detach()
                p_mean = y_soft.mean(dim=0)
                eps = 1e-8
                eff = torch.exp(-(p_mean * (p_mean + eps).log()).sum()).item()
                writer.add_scalar('train/router/effective_codes', eff, global_step)
                ys_clamped = torch.clamp(y_soft, min=eps)
                ent_per = (-ys_clamped * ys_clamped.log()).sum(dim=-1)
                eff_sample = torch.exp(ent_per).mean().item()
                writer.add_scalar('train/router/effective_codes_sample', eff_sample, global_step)
                # 专家负载均衡
                if 'exp_balance' in losses:
                    writer.add_scalar('train/exp_balance', float(losses['exp_balance'].detach().cpu()), global_step)
                print(f"step {global_step}: loss={float(loss_total.detach().cpu()):.4f} base={float(losses['loss'].detach().cpu()):.4f}"
                      f" recon={float(losses['recon'].detach().cpu()):.4f}"
                      f" proto_recon={float(losses['proto_recon'].detach().cpu()):.4f}"
                      f" delta_res={float(losses['delta_res'].detach().cpu()):.4f}"
                      f" orth={float(losses['orth'].detach().cpu()):.4f}"
                      f" vq={float(losses['vq'].detach().cpu()):.4f}"
                      f" commit={float(losses['commit'].detach().cpu()):.4f}"
                      f" kl={float(losses['kl'].detach().cpu()):.4f}"
                      f" ce={float(losses['ce'].detach().cpu()):.4f}"
                      f" ent={float(losses['ent'].detach().cpu()):.4f}"
                      f" balance={float(losses['balance'].detach().cpu()):.4f}"
                      f" exp_balance={float(losses.get('exp_balance', torch.tensor(0.0)).detach().cpu()):.4f}"
                      f" mmm={float(losses.get('mmm', torch.tensor(0.0)).detach().cpu()):.4f}"
                      f" graph={float(losses.get('graph', torch.tensor(0.0)).detach().cpu()):.4f}")

            global_step += 1

            # 按步评估（可选）并对齐 global_step 写入
            if eval_every_steps > 0 and (global_step % eval_every_steps == 0):
                metrics = eval_epoch(model, dl_val, device, cfg['loss'])
                writer.add_scalar('val/loss', metrics['loss'], global_step)
                writer.add_scalar('val/recon', metrics['recon'], global_step)
                writer.add_scalar('val/proto_recon', metrics['proto_recon'], global_step)
                writer.add_scalar('val/delta_res', metrics['delta_res'], global_step)
                writer.add_scalar('val/orth', metrics['orth'], global_step)
                writer.add_scalar('val/vq', metrics['vq'], global_step)
                writer.add_scalar('val/commit', metrics['commit'], global_step)
                print(f"[step-eval] step {global_step} eval: loss={metrics['loss']:.4f}"
                      f" recon={metrics['recon']:.4f} proto_recon={metrics['proto_recon']:.4f}"
                      f" delta_res={metrics['delta_res']:.4f} orth={metrics['orth']:.4f}"
                      f" vq={metrics['vq']:.4f} commit={metrics['commit']:.4f}")

        # Eval per epoch
        if (epoch % eval_every) == 0:
            metrics = eval_epoch(model, dl_val, device, cfg['loss'])
            # 用当前 global_step 写入，便于与训练曲线对齐
            writer.add_scalar('val/loss', metrics['loss'], global_step)
            writer.add_scalar('val/recon', metrics['recon'], global_step)
            writer.add_scalar('val/proto_recon', metrics['proto_recon'], global_step)
            writer.add_scalar('val/delta_res', metrics['delta_res'], global_step)
            writer.add_scalar('val/orth', metrics['orth'], global_step)
            writer.add_scalar('val/vq', metrics['vq'], global_step)
            writer.add_scalar('val/commit', metrics['commit'], global_step)
            writer.add_scalar('val/kl', metrics['kl'], global_step)
            writer.add_scalar('val/ce', metrics['ce'], global_step)
            writer.add_scalar('val/ent', metrics['ent'], global_step)
            writer.add_scalar('val/balance', metrics['balance'], global_step)
            print(f"Epoch {epoch} eval: loss={metrics['loss']:.4f}"
                  f" recon={metrics['recon']:.4f} proto_recon={metrics['proto_recon']:.4f}"
                  f" delta_res={metrics['delta_res']:.4f} orth={metrics['orth']:.4f}"
                  f" vq={metrics['vq']:.4f} commit={metrics['commit']:.4f}"
                  f" kl={metrics['kl']:.4f} ce={metrics['ce']:.4f} ent={metrics['ent']:.4f}")
            # 单一路径：不使用 teacher_auto 或校准阶段

        # Step scheduler
        if scheduler is not None:
            if sch_name == 'plateau':
                # Use last available metrics if we computed this epoch; otherwise evaluate quickly
                if (epoch % eval_every) == 0:
                    scheduler.step(metrics['loss'])
                else:
                    cur = eval_epoch(model, dl_val, device, cfg['loss'])
                    scheduler.step(cur['loss'])
            else:
                scheduler.step()

        # Save checkpoints
        last_path = os.path.join(run_dir, 'last.ckpt')
        _save_ckpt(last_path, model, optim, scheduler, epoch, global_step, best_val, scaler if use_amp else None)
        if (epoch % save_every) == 0 and not save_best_only:
            epoch_path = os.path.join(run_dir, f'epoch_{epoch}.ckpt')
            _save_ckpt(epoch_path, model, optim, scheduler, epoch, global_step, best_val, scaler if use_amp else None)

        if (epoch % eval_every) == 0:
            cur_val = metrics['loss']
            improved = (best_val - cur_val) > min_delta
            if improved:
                best_val = cur_val
                epochs_no_improve = 0
                best_path = os.path.join(run_dir, 'best.ckpt')
                _save_ckpt(best_path, model, optim, scheduler, epoch, global_step, best_val, scaler if use_amp else None)
            else:
                epochs_no_improve += 1

            if early_patience > 0 and epochs_no_improve >= early_patience:
                writer.add_text('early_stop', f'stopped at epoch {epoch} without improvement >= {min_delta} for {early_patience} evals', epoch)
                break

    writer.close()


if __name__ == '__main__':
    main()
