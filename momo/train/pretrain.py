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
    sums = {k: 0.0 for k in ['loss', 'recon', 'vq', 'commit', 'kl', 'ce', 'ent', 'balance', 'vq_teacher', 'recon_teacher', 'teacher_align']}
    with torch.no_grad():
        for batch in dl:
            batch = batch.to(device)
            z_hat, h_motif, e_k, logits, topk, router_info = model(batch)
            z_gt = batch.motif_target
            out = compute_losses(
                z_hat=z_hat, z_gt=z_gt, h_motif_2d=h_motif, e_k=e_k, weights=loss_cfg,
                router=router_info,
                teacher_z=router_info.get('teacher_z') if router_info is not None else None
            )
            bs = z_hat.size(0)
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
        z3d_dim=cfg['model']['z3d_dim'],
        max_atomic_num=cfg['dataset']['max_atomic_num'],
        require_pos=bool(cfg.get('teacher', {}).get('enabled', False)),
    )
    print(f"Loaded dataset from {cfg['dataset']['preprocessed_path']} with {len(ds)} molecules")

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

    # Model & optim (optionally initialize codebook by k-means on z3d)
    init_cfg = cfg['model'].get('codebook_init', {})
    init_name = str(init_cfg.get('name', 'random')).lower()
    kmeans_centroids = None
    if init_name == 'kmeans':
        sample_size = int(init_cfg.get('sample_size', 200000))
        max_iters = int(init_cfg.get('max_iters', 30))
        tol = float(init_cfg.get('tol', 1e-4))
        seed = int(cfg['misc'].get('seed', 42))
        print(f"[Init] Collecting up to {sample_size} z3d samples for k-means...")
        X = _reservoir_sample_z3d(ds, sample_size, int(cfg['model']['z3d_dim']), seed)
        K = int(cfg['model']['codebook_size'])
        N_eff = X.size(0)
        print(f"[Init] Running k-means on {N_eff} samples into {K} clusters (iters={max_iters}, tol={tol})")
        kmeans_centroids = _kmeans_torch(X, K=K, max_iters=max_iters, tol=tol, seed=seed)
        # Optional: compute inertia for logging
        with torch.no_grad():
            batch = 8192
            sse = 0.0
            for i in range(0, N_eff, batch):
                xb = X[i:i+batch]
                x2 = (xb * xb).sum(dim=1, keepdim=True)
                c2 = (kmeans_centroids * kmeans_centroids).sum(dim=1).view(1, -1)
                d2 = x2 + c2 - 2.0 * xb.matmul(kmeans_centroids.t())
                sse += float(d2.min(dim=1).values.sum().item())
        print(f"[Init] k-means inertia (sum of squared distances): {sse:.4f}")

    model = MotifVQMoE(cfg).to(device)
    # 将数据的 z3d 归一化参数注入模型，用于对齐 teacher 输出空间
    with torch.no_grad():
        model.set_z3d_norm(ds.z3d_mean.to(model.codebook.codes.device), ds.z3d_std.to(model.codebook.codes.device))
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
    if kmeans_centroids is not None:
        assert kmeans_centroids.shape == model.codebook.codes.data.shape
        with torch.no_grad():
            model.codebook.codes.data.copy_(kmeans_centroids.to(model.codebook.codes.data.dtype))
        print("[Init] Codebook initialized from k-means centroids.")
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

    # Teacher 自动权重调度（align_gate 或 linear）
    loss_base = dict(cfg['loss'])
    ta_auto = dict(cfg['loss'].get('teacher_auto', {}))
    ta_mode = str(ta_auto.get('mode', 'align_gate')).lower()
    ta_threshold = float(ta_auto.get('align_threshold', 0.25))
    ta_patience = int(ta_auto.get('patience', 3))
    ta_ramp = int(ta_auto.get('ramp_steps', 100000))
    ta_start_after = int(ta_auto.get('start_after_step', 50000))
    ta_vq_max = float(ta_auto.get('vq_max', 0.1))
    ta_recon_max = float(ta_auto.get('recon_max', 0.05))
    ta_gate_hits = 0
    ta_triggered = False
    ta_ramp_start = -1

    # Stage A：仅校准 teacher_proj 与 z_gt 的对齐，其它损失全部关闭
    calib_enabled = bool(cfg.get('teacher', {}).get('enabled', False)) and bool(cfg.get('teacher', {}).get('proj_trainable', True))
    calib_active = calib_enabled
    calib_hits = 0
    calib_min_steps = int(ta_auto.get('start_after_step', 50000))
    calib_threshold = float(ta_auto.get('align_threshold', 0.25))
    calib_patience = int(ta_auto.get('patience', 3))

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
        vq_w, rc_w = 0.0, 0.0
        # 校准阶段不再硬关主损失：保留主任务，蒸馏项仍为 0
        if calib_active:
            # 主任务保持开启
            w['recon_weight'] = float(loss_base.get('recon_weight', 1.0))
            w['vq_weight'] = float(loss_base.get('vq_weight', 1.0))
            # commit 在校准期也按预设线性升温，避免早期强绑定
            if commit_ramp_steps > 0:
                p_commit = min(1.0, float(step) / float(commit_ramp_steps))
                w['commit_weight'] = float(loss_base.get('commit_weight', 0.0)) * p_commit
            else:
                w['commit_weight'] = float(loss_base.get('commit_weight', 0.0))
            # 校准阶段：关闭 KL（避免未对齐教师牵制 y_soft）
            w['kl_weight'] = 0.0
            w['ce_weight'] = float(loss_base.get('ce_weight', 0.0))
            # 对探索相关权重增加“后期退火”调度
            ent_base = float(loss_base.get('ent_weight', 0.0))
            bal_base = float(loss_base.get('balance_weight', 0.0))
            w['ent_weight'] = _apply_weight_schedule(
                ent_base, cfg['loss'].get('ent_schedule', {}), step
            )
            w['balance_weight'] = _apply_weight_schedule(
                bal_base, cfg['loss'].get('balance_schedule', {}), step
            )
            w['vq_teacher_weight'] = 0.0
            w['recon_teacher_weight'] = 0.0
            return w, vq_w, rc_w
        if ta_mode == 'align_gate' and ta_triggered:
            if ta_ramp > 0:
                p = min(1.0, max(0.0, float(step - ta_ramp_start) / float(ta_ramp)))
            else:
                p = 1.0
            vq_w = ta_vq_max * p
            rc_w = ta_recon_max * p
        elif ta_mode == 'linear':
            start = int(ta_auto.get('start_step', 100000))
            dur = int(ta_auto.get('duration', 100000))
            if step >= start:
                p = min(1.0, max(0.0, float(step - start) / float(max(1, dur))))
                vq_w = ta_vq_max * p
                rc_w = ta_recon_max * p
        # commit 权重升温（仅在非校准阶段）
        if commit_ramp_steps > 0:
            p_commit = min(1.0, float(step) / float(commit_ramp_steps))
            w['commit_weight'] = float(loss_base.get('commit_weight', 0.0)) * p_commit
        # teacher 蒸馏两项（若启用）
        w['vq_teacher_weight'] = float(vq_w)
        w['recon_teacher_weight'] = float(rc_w)
        # 后期退火：熵与均衡
        ent_base = float(loss_base.get('ent_weight', 0.0))
        bal_base = float(loss_base.get('balance_weight', 0.0))
        w['ent_weight'] = _apply_weight_schedule(ent_base, cfg['loss'].get('ent_schedule', {}), step)
        w['balance_weight'] = _apply_weight_schedule(bal_base, cfg['loss'].get('balance_schedule', {}), step)
        return w, vq_w, rc_w

    # 路由调度参数（必须配置，不做静默回退）
    rcfg = cfg.get('router', None)
    assert rcfg is not None, 'router 配置缺失'
    tau_r_start = float(rcfg['gumbel_tau_start'])
    tau_r_end = float(rcfg['gumbel_tau_end'])
    beta_start = float(rcfg['beta_start'])
    beta_end = float(rcfg['beta_end'])
    warmup_steps = int(rcfg['warmup_steps'])
    topk_warmup_steps = int(rcfg.get('topk_warmup_steps', 0))
    # 可选：教师温度退火（高→低），以及晚期将 β 提升到更大值
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
        safeguard_until_step = -1
        for it, batch in enumerate(dl_train, start=1):
            batch = batch.to(device)
            # 线性调度 tau_r 与 beta（前 warmup_steps 过渡，之后恒定为 end）
            # 一阶段：从 tau_r_start 线性过渡到 tau_r_end
            progress = min(1.0, float(global_step) / max(1, warmup_steps))
            cur_tau_r = tau_r_start + (tau_r_end - tau_r_start) * progress
            # 二阶段（可选）：在后期进一步从当前温度线性过渡到 boost_end_tau
            if boost_start >= 0 and boost_dur > 0:
                p2 = (float(global_step) - float(boost_start)) / float(max(1, boost_dur))
                p2 = 0.0 if p2 < 0.0 else (1.0 if p2 > 1.0 else p2)
                cur_tau_r = cur_tau_r + (boost_end_tau - cur_tau_r) * p2
            cur_beta = beta_start + (beta_end - beta_start) * progress
            # 晚期将 β 进一步提升，逐渐更依赖路由器（避免长期过度均匀）
            if beta_boost_start >= 0 and beta_boost_dur > 0:
                p_b = (float(global_step) - float(beta_boost_start)) / float(max(1, beta_boost_dur))
                p_b = 0.0 if p_b < 0.0 else (1.0 if p_b > 1.0 else p_b)
                cur_beta = cur_beta + (beta_boost_end - cur_beta) * p_b
            model.set_router_params(cur_tau_r, cur_beta)
            # Safeguard: if previously flagged, temporarily soften routing
            if 'safeguard_until_step' in locals() and global_step < safeguard_until_step:
                # clamp beta and use softer selection
                cur_beta = min(cur_beta, 0.6)
                model.set_router_params(cur_tau_r, cur_beta)
                # also raise teacher temperature a bit if available
                if hasattr(model, 'teacher_temperature'):
                    try:
                        model.set_teacher_temperature(max(float(getattr(model,'teacher_temperature',2.6)), 2.8))
                    except Exception:
                        pass

            # 教师温度退火：从较高温度逐步降低到更“冷”的几何原型
            if tt_decay_steps > 0 and tt_decay_start >= 0:
                p_tt = (float(global_step) - float(tt_decay_start)) / float(max(1, tt_decay_steps))
                p_tt = 0.0 if p_tt < 0.0 else (1.0 if p_tt > 1.0 else p_tt)
                cur_tt = tt_start + (tt_end - tt_start) * p_tt
                model.set_teacher_temperature(cur_tt)
            # Top-K 分段调度（优先使用可选 schedule；否则退回原 warmup 逻辑）
            tks = rcfg.get('topk_schedule', None)
            if isinstance(tks, dict):
                s1 = int(tks.get('stage1_until', topk_warmup_steps))
                s2 = int(tks.get('stage2_until', s1))
                k2 = int(tks.get('stage2_k', 4))
                kf = int(tks.get('final_k', int(rcfg.get('topk', 1))))
                if global_step < s1:
                    model.set_topk(0)
                elif global_step < s2:
                    model.set_topk(k2)
                else:
                    model.set_topk(kf)
            else:
                # 旧逻辑：仅 warmup 后启用固定 Top‑K
                if topk_warmup_steps > 0 and global_step < topk_warmup_steps:
                    model.set_topk(0)
                else:
                    model.set_topk(int(rcfg.get('topk', 1)))
            # 校准时禁用 teacher_z 构造 p_t；对齐后再启用
            try:
                model.set_use_teacher_for_pt(not calib_active)
            except Exception:
                pass
            # Safeguard forces soft selection (expectation) for a short window
            if 'safeguard_until_step' in locals() and global_step < safeguard_until_step:
                model.set_topk(0)
            # 当前 teacher 动态权重
            weights_cur, vq_w_cur, rc_w_cur = current_weights(global_step)
            with torch.cuda.amp.autocast(enabled=use_amp):
                z_hat, h_motif, e_k, logits, topk, router_info = model(batch)
                z_gt = batch.motif_target
                losses = compute_losses(
                    z_hat=z_hat, z_gt=z_gt, h_motif_2d=h_motif, e_k=e_k,
                    weights=weights_cur, router=router_info,
                    teacher_z=router_info.get('teacher_z') if router_info is not None else None
                )
                loss = losses['loss']

            # NaN/Inf 监控与保护
            if not torch.isfinite(loss):
                msg = f"Non-finite loss detected at step {global_step}: {float(loss.detach().cpu())}"
                print(msg)
                writer.add_text('alerts/non_finite_loss', msg, global_step)
                nan_ckpt = os.path.join(run_dir, f'nan_detected_step{global_step}.ckpt')
                _save_ckpt(nan_ckpt, model, optim, None, epoch, global_step, best_val, scaler if use_amp else None)
                break

            optim.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                # 先反缩放再裁剪
                if grad_clip_norm and grad_clip_norm > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optim.step()

            # TensorBoard logging (train)
            if global_step % log_every == 0:
                writer.add_scalar('train/loss', float(losses['loss'].detach().cpu()), global_step)
                writer.add_scalar('train/recon', float(losses['recon'].detach().cpu()), global_step)
                writer.add_scalar('train/vq', float(losses['vq'].detach().cpu()), global_step)
                writer.add_scalar('train/commit', float(losses['commit'].detach().cpu()), global_step)
                writer.add_scalar('train/lr', optim.param_groups[0]['lr'], global_step)
                writer.add_scalar('train/kl', float(losses['kl'].detach().cpu()), global_step)
                writer.add_scalar('train/ce', float(losses['ce'].detach().cpu()), global_step)
                writer.add_scalar('train/ent', float(losses['ent'].detach().cpu()), global_step)
                writer.add_scalar('train/router_tau_r', cur_tau_r, global_step)
                writer.add_scalar('train/router_beta', cur_beta, global_step)
                writer.add_scalar('train/teacher_auto/vq_weight', float(vq_w_cur), global_step)
                writer.add_scalar('train/teacher_auto/recon_weight', float(rc_w_cur), global_step)
                # 代码使用率与 Top-k 命中率
                if model.router_topk > 1 and 'topk_indices' in router_info:
                    sel = router_info['topk_indices'].detach()
                    sel_flat = sel.reshape(-1)
                    nn_idx = router_info.get('teacher_nn_index', None)
                    hit = 0.0
                    if nn_idx is not None:
                        nn_idx = nn_idx.detach()
                        hit = (sel == nn_idx.view(-1, 1)).any(dim=1).float().mean().item()
                    writer.add_scalar('train/router/topk_nn_hit', hit, global_step)
                    used = torch.unique(sel_flat).numel() / float(model.codebook_size)
                    writer.add_scalar('train/router/usage_unique_ratio', used, global_step)
                    writer.add_histogram('train/router/selected_codes', sel_flat.to(torch.float32), global_step)
                else:
                    sel = topk.view(-1) if 'topk' in locals() else topk_idx.view(-1)
                    nn_idx = router_info.get('teacher_nn_index', None)
                    hit = 0.0
                    if nn_idx is not None:
                        nn_idx = nn_idx.detach()
                        hit = (sel == nn_idx).float().mean().item()
                    writer.add_scalar('train/router/top1_nn_hit', hit, global_step)
                    used = torch.unique(sel).numel() / float(model.codebook_size)
                    writer.add_scalar('train/router/usage_unique_ratio', used, global_step)
                    writer.add_histogram('train/router/selected_codes', sel.to(torch.float32), global_step)
                # 有效代码数（两种口径：batch-mean 与 sample-mean）
                y_soft = router_info['y_soft'].detach()
                p_t = router_info.get('p_t', None)
                p_mean = y_soft.mean(dim=0)
                eps = 1e-8
                eff = torch.exp(-(p_mean * (p_mean + eps).log()).sum()).item()
                writer.add_scalar('train/router/effective_codes', eff, global_step)
                ys_clamped = torch.clamp(y_soft, min=eps)
                ent_per = (-ys_clamped * ys_clamped.log()).sum(dim=-1)
                eff_sample = torch.exp(ent_per).mean().item()
                writer.add_scalar('train/router/effective_codes_sample', eff_sample, global_step)
                # Write safeguard flag
                writer.add_scalar('train/router/safeguard_active', 1.0 if ('safeguard_until_step' in locals() and global_step < safeguard_until_step) else 0.0, global_step)
                # 额外监控：与教师分布的匹配度
                if p_t is not None:
                    pt = p_t.detach()
                    ys = torch.clamp(y_soft, min=eps)
                    ptc = torch.clamp(pt, min=eps)
                    kl1 = torch.sum(ptc * (ptc.log() - ys.log()), dim=-1).mean().item()
                    kl2 = torch.sum(ys * (ys.log() - ptc.log()), dim=-1).mean().item()
                    writer.add_scalar('train/router/kl_to_teacher', kl1 + kl2, global_step)
                    writer.add_scalar('train/router/y_soft_entropy', float((-ys * ys.log()).sum(dim=-1).mean().item()), global_step)
                    writer.add_scalar('train/router/p_t_entropy', float((-ptc * ptc.log()).sum(dim=-1).mean().item()), global_step)
                    # Trigger safeguard with hysteresis: BOTH low entropy and low effective codes (sample-wise)
                    try:
                        ent_val = float((-ys * ys.log()).sum(dim=-1).mean().item())
                    except Exception:
                        ent_val = 0.0
                    bad = (ent_val < 0.05) and (eff_sample < 8.0)
                    # Initialize counters if missing
                    if 'sg_bad_count' not in locals():
                        sg_bad_count = 0
                    if 'sg_good_count' not in locals():
                        sg_good_count = 0
                    # Activate only when not currently active and we saw enough consecutive bad signals
                    if ('safeguard_until_step' not in locals()) or (global_step >= safeguard_until_step):
                        if bad:
                            sg_bad_count += 1
                            sg_good_count = 0
                        else:
                            sg_good_count += 1
                            sg_bad_count = 0
                        if bad and sg_bad_count >= 5:
                            safeguard_until_step = global_step + 2000
                    else:
                        # During safeguard window, allow early release after consecutive good observations
                        if not bad:
                            sg_good_count += 1
                            if sg_good_count >= 5:
                                safeguard_until_step = global_step
                if 'vq_teacher' in losses:
                    writer.add_scalar('train/vq_teacher', float(losses['vq_teacher'].detach().cpu()), global_step)
                    writer.add_scalar('train/recon_teacher', float(losses['recon_teacher'].detach().cpu()), global_step)
                    writer.add_scalar('train/teacher_align', float(losses['teacher_align'].detach().cpu()), global_step)
                if 'balance' in losses:
                    writer.add_scalar('train/router/balance', float(losses['balance'].detach().cpu()), global_step)
                print(f"step {global_step}: loss={float(losses['loss'].detach().cpu()):.4f}"
                      f" recon={float(losses['recon'].detach().cpu()):.4f}"
                      f" vq={float(losses['vq'].detach().cpu()):.4f}"
                      f" commit={float(losses['commit'].detach().cpu()):.4f}"
                      f" kl={float(losses['kl'].detach().cpu()):.4f}"
                      f" ce={float(losses['ce'].detach().cpu()):.4f}"
                      f" ent={float(losses['ent'].detach().cpu()):.4f}"
                      f" balance={float(losses['balance'].detach().cpu()):.4f}")

            global_step += 1

            # 按步评估（可选）并对齐 global_step 写入
            if eval_every_steps > 0 and (global_step % eval_every_steps == 0):
                w_eval, _, _ = current_weights(global_step)
                metrics = eval_epoch(model, dl_val, device, w_eval)
                writer.add_scalar('val/loss', metrics['loss'], global_step)
                writer.add_scalar('val/recon', metrics['recon'], global_step)
                writer.add_scalar('val/vq', metrics['vq'], global_step)
                writer.add_scalar('val/commit', metrics['commit'], global_step)
                writer.add_scalar('val/teacher_align', metrics['teacher_align'], global_step)
                print(f"[step-eval] step {global_step} eval: loss={metrics['loss']:.4f}"
                      f" recon={metrics['recon']:.4f} vq={metrics['vq']:.4f} commit={metrics['commit']:.4f}")
                # 自动开启门控：对齐稳定后触发蒸馏权重升温
                if ta_mode == 'align_gate' and (not ta_triggered) and global_step >= ta_start_after:
                    if metrics['teacher_align'] <= ta_threshold:
                        ta_gate_hits += 1
                    else:
                        ta_gate_hits = 0
                    if ta_gate_hits >= ta_patience:
                        ta_triggered = True
                        ta_ramp_start = global_step
                        writer.add_text('teacher_auto', f'triggered at step {global_step}, ramp_steps={ta_ramp}', global_step)
                # 结束校准阶段：teacher 对齐稳定
                if calib_active and global_step >= calib_min_steps:
                    if metrics['teacher_align'] <= calib_threshold:
                        calib_hits += 1
                    else:
                        calib_hits = 0
                    if calib_hits >= calib_patience:
                        calib_active = False
                        writer.add_text('calibration', f'calibration finished at step {global_step}', global_step)

        # Eval per epoch
        if (epoch % eval_every) == 0:
            w_eval, _, _ = current_weights(global_step)
            metrics = eval_epoch(model, dl_val, device, w_eval)
            # 用当前 global_step 写入，便于与训练曲线对齐
            writer.add_scalar('val/loss', metrics['loss'], global_step)
            writer.add_scalar('val/recon', metrics['recon'], global_step)
            writer.add_scalar('val/vq', metrics['vq'], global_step)
            writer.add_scalar('val/commit', metrics['commit'], global_step)
            writer.add_scalar('val/kl', metrics['kl'], global_step)
            writer.add_scalar('val/ce', metrics['ce'], global_step)
            writer.add_scalar('val/ent', metrics['ent'], global_step)
            writer.add_scalar('val/balance', metrics['balance'], global_step)
            writer.add_scalar('val/vq_teacher', metrics['vq_teacher'], global_step)
            writer.add_scalar('val/recon_teacher', metrics['recon_teacher'], global_step)
            writer.add_scalar('val/teacher_align', metrics['teacher_align'], global_step)
            print(f"Epoch {epoch} eval: loss={metrics['loss']:.4f}"
                  f" recon={metrics['recon']:.4f} vq={metrics['vq']:.4f} commit={metrics['commit']:.4f}"
                  f" kl={metrics['kl']:.4f} ce={metrics['ce']:.4f} ent={metrics['ent']:.4f}"
                  f" vqt={metrics['vq_teacher']:.4f} rct={metrics['recon_teacher']:.4f}"
                  f" talign={metrics['teacher_align']:.4f}")
            if ta_mode == 'align_gate' and (not ta_triggered) and global_step >= ta_start_after:
                if metrics['teacher_align'] <= ta_threshold:
                    ta_gate_hits += 1
                else:
                    ta_gate_hits = 0
                if ta_gate_hits >= ta_patience:
                    ta_triggered = True
                    ta_ramp_start = global_step
                    writer.add_text('teacher_auto', f'triggered at step {global_step}, ramp_steps={ta_ramp}', global_step)
            # 结束校准阶段（按 epoch 评估）
            if calib_active and global_step >= calib_min_steps:
                if metrics['teacher_align'] <= calib_threshold:
                    calib_hits += 1
                else:
                    calib_hits = 0
                if calib_hits >= calib_patience:
                    calib_active = False
                    writer.add_text('calibration', f'calibration finished at step {global_step}', global_step)

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
