# momo/train/pretrain.py
import os
import argparse
from typing import Dict, Optional, List
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


def _scan_dataset_stats(ds: PCQM4Mv2MotifDataset) -> tuple[int, int, List[bool]]:
    """Scan dataset once to collect startup diagnostics.

    Returns:
      max_motif_type_id: maximum motif type id found in records
      zero_edge_total: number of molecules with zero motif edges
      zero_edge_flags: per-record flag for fast split-level stats
    """
    max_id = 0
    zero_total = 0
    zero_flags: List[bool] = []

    for rec in ds.data:  # type: ignore[attr-defined]
        mei = rec['motif_edge_index']
        assert isinstance(mei, (list, tuple)) and len(mei) == 2, "motif_edge_index must be a 2-row list"
        e = len(mei[0])
        is_zero = (e == 0)
        zero_flags.append(is_zero)
        if is_zero:
            zero_total += 1

        arr = rec['motif_type_id']
        cur = max(arr) if arr else 0
        if cur > max_id:
            max_id = int(cur)

    return int(max_id), int(zero_total), zero_flags


def _check_neighbor_type_range(batch, vocab_size: int, where: str) -> None:
    nb = batch.motif_edge_neighbor_type
    if nb.numel() == 0:
        return

    nb_cpu = nb.long().view(-1).cpu()
    nb_min = int(nb_cpu.min().item())
    nb_max = int(nb_cpu.max().item())
    bad = (nb_cpu < 0) | (nb_cpu >= int(vocab_size))
    if not bad.any():
        return

    bad_vals = nb_cpu[bad]
    bad_preview = bad_vals[:16].tolist()
    parts: List[str] = [
        f"CPU check failed {where}: motif_edge_neighbor_type out of range",
        f"min_id={nb_min}",
        f"max_id={nb_max}",
        f"vocab_size={int(vocab_size)}",
        f"bad_count={int(bad.sum().item())}",
        f"bad_min={int(bad_vals.min().item())}",
        f"bad_max={int(bad_vals.max().item())}",
        f"bad_preview={bad_preview}",
    ]

    assert hasattr(batch, "rec_idx") and hasattr(batch, "num_motifs") and hasattr(batch, "motif_edge_index")
    rec_idx = batch.rec_idx.view(-1).long().cpu()
    num_motifs = batch.num_motifs.view(-1).long().cpu()
    mei = batch.motif_edge_index
    assert mei.dim() == 2 and mei.size(0) == 2 and mei.size(1) == nb_cpu.numel()
    assert rec_idx.numel() == num_motifs.numel() and num_motifs.numel() > 0
    offsets = torch.cumsum(
        torch.cat([torch.zeros(1, dtype=torch.long), num_motifs], dim=0),
        dim=0,
    )
    src = mei[0].long().view(-1).cpu()
    edge_graph = torch.bucketize(src, offsets[1:], right=True)
    bad_graph = torch.unique(edge_graph[bad], sorted=True)
    bad_rec = rec_idx[bad_graph]
    parts.append(f"bad_rec_idx={bad_rec[:16].tolist()}")

    raise AssertionError(" | ".join(parts))


def _collect_code_init_latents(model: MotifVQMoE, dl: DataLoader, sample_size: int, device: str) -> torch.Tensor:
    """Collect up to sample_size latent targets across batches for k-means init.
    在 CPU 上先检查 batch 后的 motif graph，再搬到 GPU，避免 CUDA 越界难定位。
    """
    zs: list[torch.Tensor] = []
    total = 0
    model.eval()

    with torch.no_grad():
        for batch in dl:
            # ---------- CPU check before batch.to(device) ----------
            assert hasattr(batch, 'num_motifs')
            assert hasattr(batch, 'motif_edge_index')
            assert hasattr(batch, 'motif_edge_attr')
            assert hasattr(batch, 'motif_edge_neighbor_type')
            assert hasattr(batch, 'motif_edge_attach_pos_src')
            assert hasattr(batch, 'motif_edge_attach_pos_dst')

            num_motifs_total = int(batch.num_motifs.sum().item())
            mei = batch.motif_edge_index  # still on CPU

            assert mei.dim() == 2 and mei.size(0) == 2, \
                f"motif_edge_index must be [2, E], got {tuple(mei.shape)}"

            E = int(mei.size(1))
            if E > 0:
                mx = int(mei.max().item())
                mn = int(mei.min().item())
                assert 0 <= mn and mx < num_motifs_total, (
                    f"CPU check failed in kmeans init: motif_edge_index out of bounds: "
                    f"min={mn}, max={mx}, num_motifs_total={num_motifs_total}, E={E}"
                )

                assert batch.motif_edge_attr.size(0) == E, (
                    f"motif_edge_attr rows {batch.motif_edge_attr.size(0)} != E {E}"
                )
                assert batch.motif_edge_neighbor_type.numel() == E, (
                    f"motif_edge_neighbor_type len {batch.motif_edge_neighbor_type.numel()} != E {E}"
                )
                assert batch.motif_edge_attach_pos_src.numel() == E, (
                    f"motif_edge_attach_pos_src len {batch.motif_edge_attach_pos_src.numel()} != E {E}"
                )
                assert batch.motif_edge_attach_pos_dst.numel() == E, (
                    f"motif_edge_attach_pos_dst len {batch.motif_edge_attach_pos_dst.numel()} != E {E}"
                )

                _check_neighbor_type_range(batch, int(model.neighbor_type_vocab), "in kmeans init")
            else:
                # empty-edge samples must still have consistent empty tensors
                assert batch.motif_edge_attr.size(0) == 0
                assert batch.motif_edge_neighbor_type.numel() == 0
                assert batch.motif_edge_attach_pos_src.numel() == 0
                assert batch.motif_edge_attach_pos_dst.numel() == 0
            # ------------------------------------------------------

            batch = batch.to(device)
            _, _, info = model(batch)
            src = info['z_template_target'] if 'z_template_target' in info else (info['z3d_target'] if 'z3d_target' in info else info['z_e'])
            z = src.detach().cpu().to(torch.float32)
            zs.append(z)
            total += z.size(0)
            if total >= sample_size:
                break

    if not zs:
        raise RuntimeError('Failed to collect latents for k-means init')

    Z = torch.cat(zs, dim=0)
    if Z.size(0) > sample_size:
        Z = Z[:sample_size]
    return Z


def _kmeans_torch(X: torch.Tensor, K: int, max_iters: int = 30, tol: float = 1e-4, seed: int = 42) -> torch.Tensor:
    """Simple k-means on CPU with mini-batch assignment to avoid O(N*K) memory explosion.

    X: [N, D] float32 CPU tensor
    Returns centroids [K, D]
    """
    assert X.device.type == 'cpu'
    N, D = X.size(0), X.size(1)
    assert N >= K, f"k-means init requires at least K samples, got N={N}, K={K}"
    g = torch.Generator(device='cpu').manual_seed(seed)
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
        empty = (counts == 0).nonzero(as_tuple=False).view(-1)
        assert empty.numel() == 0, f"k-means produced empty clusters: {empty.tolist()}"

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
    sums = {k: 0.0 for k in ['loss', 'edge', 'main_3d', 'template', 'vq', 'commit', 'orth', 'delta_l2', 'ctx_delta_l2', 'ctx_delta_orth']}
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
        z3d_dim=int(cfg['model'].get('edge_target_dim', 7)),
        max_atomic_num=cfg['dataset']['max_atomic_num'],
        require_pos=bool(cfg.get('teacher', {}).get('enabled', False)),
    )
    print(f"Loaded dataset from {cfg['dataset']['preprocessed_path']} with {len(ds)} molecules")
    # Single-pass startup scan: max motif type id + zero-edge flags.
    max_motif_type_id = 0
    zero_flags: List[bool] = []
    total = len(ds)
    max_motif_type_id, zero_total, zero_flags = _scan_dataset_stats(ds)
    pct = (100.0 * zero_total / max(1, total))
    print(f"Zero-motif-edge molecules: {zero_total}/{total} ({pct:.2f}%)")
    assert int(ds.edge_target_dim) == int(cfg['model'].get('edge_target_dim', 7)), (
        f"edge dims mismatch: dataset={int(ds.edge_target_dim)} vs cfg={int(cfg['model'].get('edge_target_dim', 7))}"
    )
    cfg_vocab = int(cfg['model']['neighbor_type_vocab_size'])
    assert cfg_vocab > max_motif_type_id, (
        f"model.neighbor_type_vocab_size ({cfg_vocab}) must be greater than "
        f"max motif_type_id in preprocessed data ({max_motif_type_id})"
    )
    val_ratio = float(cfg['dataset'].get('val_ratio', 0.05))
    n_total = len(ds)
    n_val = max(1, int(round(n_total * val_ratio)))
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(int(cfg['misc'].get('seed', 42)))
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=gen)
    # Per-split zero-edge ratio
    ti = train_ds.indices  # type: ignore[attr-defined]
    vi = val_ds.indices    # type: ignore[attr-defined]
    assert len(zero_flags) == len(ds)
    zero_train = sum(1 for i in ti if zero_flags[i])
    zero_val = sum(1 for i in vi if zero_flags[i])
    pct_tr = (100.0 * zero_train / max(1, len(ti)))
    pct_va = (100.0 * zero_val / max(1, len(vi)))
    print(f"Zero-motif-edge (train): {zero_train}/{len(ti)} ({pct_tr:.2f}%)")
    print(f"Zero-motif-edge (val):   {zero_val}/{len(vi)} ({pct_va:.2f}%)")

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
    assert bool(cfg.get('teacher', {}).get('enabled', False)), 'teacher.enabled must be true for main 3D supervision'

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
    # Codebook 初始化：对目标几何 latent（teacher）或 encoder latent 做 k-means。
    if init_name == 'kmeans':
        sample_size = int(init_cfg.get('sample_size', 200000))
        max_iters = int(init_cfg.get('max_iters', 30))
        tol = float(init_cfg.get('tol', 1e-4))
        seed = int(cfg['misc'].get('seed', 42))
        print(f"[Init] Collecting up to {sample_size} latent targets for k-means...")
        Z = _collect_code_init_latents(model, dl_train, sample_size, device)
        
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

    # 基本权重与调度（仅保留 commit 升温）
    loss_base = dict(cfg['loss'])
    commit_ramp_steps = int(cfg.get('loss', {}).get('commit_ramp_steps', 50000))

    def current_weights(step: int):
        w = dict(loss_base)
        if commit_ramp_steps > 0:
            p_commit = min(1.0, float(step) / float(commit_ramp_steps))
            w['commit_weight'] = float(loss_base.get('commit_weight', 0.0)) * p_commit
        return w

    # 路由调度参数（必须配置，不做静默回退）
    rcfg = cfg.get('router', None)
    assert rcfg is not None, 'router 配置缺失'
    tau_r_start = float(rcfg['gumbel_tau_start'])
    tau_r_end = float(rcfg['gumbel_tau_end'])
    warmup_steps = int(rcfg['warmup_steps'])
    # 可选：教师温度退火（高→低），以及晚期将 β 提升到更大值（保留，但不强制要求 teacher）
    tt_start = float(rcfg.get('teacher_temperature_start', rcfg.get('teacher_temperature', 1.0)))
    tt_end = float(rcfg.get('teacher_temperature_end', tt_start))
    tt_decay_start = int(rcfg.get('teacher_temp_decay_start', -1))
    tt_decay_steps = int(rcfg.get('teacher_temp_decay_steps', 0))

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
        # for it, batch in enumerate(dl_train, start=1):
        #     batch = batch.to(device)
        for it, batch in enumerate(dl_train, start=1):
            # 在 CPU 上检查 batched motif graph
            num_motifs_total = int(batch.num_motifs.sum().item())
            mei = batch.motif_edge_index
            if mei.numel() > 0:
                mx = int(mei.max().item())
                mn = int(mei.min().item())
                E = int(mei.size(1))
                assert 0 <= mn and mx < num_motifs_total, (
                    f"CPU check failed: motif_edge_index out of bounds: "
                    f"min={mn}, max={mx}, num_motifs_total={num_motifs_total}, E={E}"
                )
                assert batch.motif_edge_attr.size(0) == E, (
                    f"motif_edge_attr rows {batch.motif_edge_attr.size(0)} != E {E}"
                )
                assert batch.motif_edge_neighbor_type.numel() == E, (
                    f"motif_edge_neighbor_type len {batch.motif_edge_neighbor_type.numel()} != E {E}"
                )
                assert batch.motif_edge_attach_pos_src.numel() == E, (
                    f"motif_edge_attach_pos_src len {batch.motif_edge_attach_pos_src.numel()} != E {E}"
                )
                assert batch.motif_edge_attach_pos_dst.numel() == E, (
                    f"motif_edge_attach_pos_dst len {batch.motif_edge_attach_pos_dst.numel()} != E {E}"
                )

            _check_neighbor_type_range(batch, int(model.neighbor_type_vocab), f"in train loop (epoch={epoch}, iter={it})")

            batch = batch.to(device)
            # 按步应用损失权重与调度（熵/均衡/commit 等）
            weights_cur = current_weights(global_step)
            # 路由温度调度（应用于模型前向）
            # 原型温度 tau_r：线性从 start→end，后期可选 boost；可被熵闭环覆盖
            if warmup_steps > 0:
                p_tau = min(1.0, max(0.0, float(global_step) / float(max(1, warmup_steps))))
            else:
                p_tau = 1.0
            tau_sched = tau_r_start + (tau_r_end - tau_r_start) * p_tau
            if boost_start >= 0 and boost_dur > 0 and global_step >= boost_start:
                q = min(1.0, max(0.0, float(global_step - boost_start) / float(max(1, boost_dur))))
                tau_sched = tau_sched + (boost_end_tau - tau_r_end) * q
            tau_apply = float(tau_sched)
            if hasattr(model, 'set_router_temperature'):
                model.set_router_temperature(tau_apply)
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
                h_motif_2d, h_enh, router_info = model(batch)
                losses = compute_losses(h_motif_2d=h_motif_2d, router=router_info, weights=weights_cur)
                loss_total = losses['loss']

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
                writer.add_scalar('train/loss', float(loss_total.detach().cpu()), global_step)
                writer.add_scalar('train/loss_base', float(losses['loss'].detach().cpu()), global_step)
                writer.add_scalar('train/edge', float(losses['edge'].detach().cpu()), global_step)
                writer.add_scalar('train/main_3d', float(losses['main_3d'].detach().cpu()), global_step)
                writer.add_scalar('train/template', float(losses['template'].detach().cpu()), global_step)
                writer.add_scalar('train/orth', float(losses['orth'].detach().cpu()), global_step)
                writer.add_scalar('train/vq', float(losses['vq'].detach().cpu()), global_step)
                writer.add_scalar('train/commit', float(losses['commit'].detach().cpu()), global_step)
                writer.add_scalar('train/delta_l2', float(losses['delta_l2'].detach().cpu()), global_step)
                writer.add_scalar('train/ctx_delta_l2', float(losses['ctx_delta_l2'].detach().cpu()), global_step)
                writer.add_scalar('train/ctx_delta_orth', float(losses['ctx_delta_orth'].detach().cpu()), global_step)
                writer.add_scalar('train/lr', optim.param_groups[0]['lr'], global_step)

                sel = router_info['code_idx'].detach()
                used = torch.unique(sel).numel() / float(model.codebook_size)
                writer.add_scalar('train/router/usage_unique_ratio', used, global_step)
                p_code_soft = router_info['p_code_soft'].detach()
                p_mean = p_code_soft.mean(dim=0)
                eps = 1e-8
                eff = torch.exp(-(p_mean * (p_mean + eps).log()).sum()).item()
                writer.add_scalar('train/router/effective_codes', eff, global_step)
                ys_clamped = torch.clamp(p_code_soft, min=eps)
                ent_per = (-ys_clamped * ys_clamped.log()).sum(dim=-1)
                eff_sample = torch.exp(ent_per).mean().item()
                writer.add_scalar('train/router/effective_codes_sample', eff_sample, global_step)
                print(f"step {global_step}: loss={float(loss_total.detach().cpu()):.4f} base={float(losses['loss'].detach().cpu()):.4f}"
                      f" edge={float(losses['edge'].detach().cpu()):.4f}"
                      f" main_3d={float(losses['main_3d'].detach().cpu()):.4f}"
                      f" template={float(losses['template'].detach().cpu()):.4f}"
                      f" orth={float(losses['orth'].detach().cpu()):.4f}"
                      f" vq={float(losses['vq'].detach().cpu()):.4f}"
                      f" commit={float(losses['commit'].detach().cpu()):.4f}"
                      f" delta_l2={float(losses['delta_l2'].detach().cpu()):.4f}"
                      f" ctx_delta_l2={float(losses['ctx_delta_l2'].detach().cpu()):.4f}"
                      f" ctx_delta_orth={float(losses['ctx_delta_orth'].detach().cpu()):.4f}")

            global_step += 1

            # 按步评估（可选）并对齐 global_step 写入
            if eval_every_steps > 0 and (global_step % eval_every_steps == 0):
                metrics = eval_epoch(model, dl_val, device, cfg['loss'])
                writer.add_scalar('val/loss', metrics['loss'], global_step)
                writer.add_scalar('val/edge', metrics['edge'], global_step)
                writer.add_scalar('val/main_3d', metrics['main_3d'], global_step)
                writer.add_scalar('val/template', metrics['template'], global_step)
                writer.add_scalar('val/orth', metrics['orth'], global_step)
                writer.add_scalar('val/vq', metrics['vq'], global_step)
                writer.add_scalar('val/commit', metrics['commit'], global_step)
                writer.add_scalar('val/ctx_delta_l2', metrics['ctx_delta_l2'], global_step)
                writer.add_scalar('val/ctx_delta_orth', metrics['ctx_delta_orth'], global_step)
                print(f"[step-eval] step {global_step} eval: loss={metrics['loss']:.4f}"
                      f" edge={metrics['edge']:.4f} main_3d={metrics['main_3d']:.4f}"
                      f" template={metrics['template']:.4f}"
                      f" orth={metrics['orth']:.4f} vq={metrics['vq']:.4f} commit={metrics['commit']:.4f}"
                      f" ctx_delta_l2={metrics['ctx_delta_l2']:.4f} ctx_delta_orth={metrics['ctx_delta_orth']:.4f}")

        # Eval per epoch
        if (epoch % eval_every) == 0:
            metrics = eval_epoch(model, dl_val, device, cfg['loss'])
            writer.add_scalar('val/loss', metrics['loss'], global_step)
            writer.add_scalar('val/edge', metrics['edge'], global_step)
            writer.add_scalar('val/main_3d', metrics['main_3d'], global_step)
            writer.add_scalar('val/template', metrics['template'], global_step)
            writer.add_scalar('val/orth', metrics['orth'], global_step)
            writer.add_scalar('val/vq', metrics['vq'], global_step)
            writer.add_scalar('val/commit', metrics['commit'], global_step)
            writer.add_scalar('val/ctx_delta_l2', metrics['ctx_delta_l2'], global_step)
            writer.add_scalar('val/ctx_delta_orth', metrics['ctx_delta_orth'], global_step)
            print(f"Epoch {epoch} eval: loss={metrics['loss']:.4f}"
                  f" edge={metrics['edge']:.4f} main_3d={metrics['main_3d']:.4f}"
                  f" template={metrics['template']:.4f}"
                  f" orth={metrics['orth']:.4f} vq={metrics['vq']:.4f} commit={metrics['commit']:.4f}"
                  f" ctx_delta_l2={metrics['ctx_delta_l2']:.4f} ctx_delta_orth={metrics['ctx_delta_orth']:.4f}")

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
