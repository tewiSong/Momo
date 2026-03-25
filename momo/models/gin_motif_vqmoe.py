from typing import Tuple, Dict

import torch
from torch import nn
import torch.nn.functional as F
from .schnet import AtomSchNet
from torch_geometric.nn import GINConv
from torch_scatter import scatter_mean


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AtomEncoder(nn.Module):
    def __init__(self, max_z: int, hidden: int):
        super().__init__()
        self.emb = nn.Embedding(max_z + 1, hidden)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        assert z.dtype == torch.long
        return self.emb(z)


class GINStack(nn.Module):
    def __init__(self, hidden: int, num_layers: int, mlp_hidden: int, dropout: float):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            mlp = MLP(hidden, mlp_hidden, hidden, dropout)
            conv = GINConv(mlp)
            layers.append(conv)
        self.layers = nn.ModuleList(layers)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.layers:
            x = conv(x, edge_index)
            x = self.act(x)
        return x


class Router(nn.Module):
    def __init__(self, in_dim: int, codebook_size: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, codebook_size)

    def forward(self, h_motif: torch.Tensor) -> torch.Tensor:
        # 返回 logits [N_motif, K]
        logits = self.fc(h_motif)
        assert logits.dim() == 2
        return logits


class Codebook(nn.Module):
    def __init__(self, codebook_size: int, code_dim: int):
        super().__init__()
        self.codes = nn.Parameter(torch.randn(codebook_size, code_dim) * 0.02)

    def lookup(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: [N_motif]
        return self.codes[idx]


class ExpertMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.mlp = MLP(in_dim, hidden, out_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MotifVQMoE(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        mcfg = cfg['model']
        dcfg = cfg['dataset']
        # 路由相关的训练超参从 cfg['router'] 获取（必须提供）
        rcfg = cfg.get('router', None)
        assert rcfg is not None, "router 配置缺失，请在配置文件加入 router 块"
        required_router_keys = ['distance_temperature', 'gumbel_tau_start', 'gumbel_tau_end', 'beta_start', 'beta_end', 'warmup_steps', 'topk']
        for k in required_router_keys:
            assert k in rcfg, f"router.{k} 缺失"
        self.hidden = int(mcfg['hidden_dim'])
        self.z3d_dim = int(mcfg['z3d_dim'])
        self.codebook_size = int(mcfg['codebook_size'])
        # 使用融合路由，支持 Top-K 选通
        self.router_topk = int(rcfg['topk'])
        assert self.router_topk >= 1

        self.atom_encoder = AtomEncoder(int(dcfg['max_atomic_num']), self.hidden)
        self.gin = GINStack(self.hidden, int(mcfg['num_gin_layers']), int(mcfg['gin_mlp_hidden']), float(mcfg['dropout']))
        self.router = Router(self.hidden, self.codebook_size)
        # 按文档，Code 原型维度与 z_3D 对齐
        self.codebook = Codebook(self.codebook_size, self.z3d_dim)
        self.expert = ExpertMLP(self.hidden + self.z3d_dim, int(mcfg['expert_mlp_hidden']), self.z3d_dim, float(mcfg['dropout']))
        # 为了在 commit 损失中将 2D 表征与 3D code 对齐，需要一个从 hidden->z3d_dim 的投影
        self.commit_proj = nn.Linear(self.hidden, self.z3d_dim)

        # Teacher (3D encoder) 配置
        tcfg = cfg.get('teacher', None)
        self.teacher_enabled = bool(tcfg and tcfg.get('enabled', False))
        # 在 teacher 对齐前，可选择不使用 teacher_z 构造 p_t（由训练循环动态门控）
        self.use_teacher_for_pt = True

        readout = str(mcfg['readout']).lower()
        assert readout in ['sum', 'attention']
        self.readout_type = readout
        if readout == 'attention':
            self.att_proj = nn.Linear(self.hidden + self.z3d_dim, 1)

        # Router 温度与融合系数（由训练循环动态设置）
        self.dist_temperature = float(rcfg['distance_temperature'])
        # 教师/几何分布温度（若未提供则复用 distance_temperature）
        self.teacher_temperature = float(rcfg.get('teacher_temperature', self.dist_temperature))
        self._tau_r = float(rcfg['gumbel_tau_start'])
        self._beta = float(rcfg['beta_start'])

        # 可选：与数据一致的 z3d 归一化参数（用于将 teacher_z 对齐到 z_gt 空间）
        self.register_buffer('z3d_mean', torch.zeros(self.z3d_dim), persistent=False)
        self.register_buffer('z3d_std', torch.ones(self.z3d_dim), persistent=False)
        self._z3d_norm_set = False

        # Teacher 模型与投影
        if self.teacher_enabled:
            th = int(tcfg['hidden_dim'])
            self.teacher = AtomSchNet(
                hidden_channels=th,
                num_filters=int(tcfg['num_filters']),
                num_interactions=int(tcfg['num_interactions']),
                num_gaussians=int(tcfg['num_gaussians']),
                cutoff=float(tcfg['cutoff'])
            )
            # 高阶聚合：在 motif 级别拼接一阶均值与二阶方差信息，再用非线性头映射到 z3d 维
            in_dim = th * 2
            self.teacher_proj = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, th),
                nn.ReLU(),
                nn.Linear(th, self.z3d_dim),
            )

    def set_router_params(self, tau_r: float, beta: float) -> None:
        self._tau_r = float(tau_r)
        self._beta = float(beta)

    # 动态调整教师分布温度（用于训练后期收敛到更“冷”的几何原型）
    def set_teacher_temperature(self, t: float) -> None:
        self.teacher_temperature = float(t)

    # 训练循环可调用：控制是否使用 teacher_z 生成 p_t（未对齐阶段关闭以避免牵制）
    def set_use_teacher_for_pt(self, use: bool) -> None:
        self.use_teacher_for_pt = bool(use)

    def set_z3d_norm(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """设置与数据一致的 z3d 归一化均值与方差，用于对齐 teacher 输出空间。
        期望形状 [D]，dtype float32/float64，device 任意（会移动到模型 device）。
        """
        m = mean.detach().to(dtype=self.z3d_mean.dtype, device=self.z3d_mean.device)
        s = std.detach().to(dtype=self.z3d_std.dtype, device=self.z3d_std.device)
        # 防止除零
        s = torch.where(s > 0, s, torch.ones_like(s))
        self.z3d_mean.copy_(m)
        self.z3d_std.copy_(s)
        self._z3d_norm_set = True

    @staticmethod
    def motif_pool(x_atom: torch.Tensor, motif_global_idx: torch.Tensor, num_global_motifs: int) -> torch.Tensor:
        assert motif_global_idx.dim() == 1 and x_atom.size(0) == motif_global_idx.size(0)
        h_motif = scatter_mean(x_atom, motif_global_idx, dim=0, dim_size=num_global_motifs)
        assert h_motif.shape[0] == num_global_motifs
        return h_motif

    def forward(self, data: 'torch_geometric.data.Batch') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # 原子编码 + GIN
        x = self.atom_encoder(data.z)
        x = self.gin(x, data.edge_index)
        assert x.dim() == 2 and x.size(1) == self.hidden

        # 全局 motif 索引
        from momo.data.pcqm4mv2 import motif_global_index, _infer_num_motifs_per_graph
        motif_gidx = motif_global_index(data.motif_id, data.batch)
        # 基于 batch 内部一致性推断每图 motif 数，避免 DataLoader 拼接歧义
        inferred_counts = _infer_num_motifs_per_graph(data.motif_id, data.batch)
        num_global_motifs = int(inferred_counts.sum().item())
        # 标签维度必须与推断总数严格一致，否则属于批处理错误
        assert num_global_motifs == int(data.motif_target.size(0)), (
            f"motif_target not batched correctly: inferred_total={num_global_motifs}, "
            f"motif_target.size(0)={int(data.motif_target.size(0))}"
        )
        # 在批次内，确保 motif_gidx 索引未越界
        max_gid = int(motif_gidx.max().item())
        min_gid = int(motif_gidx.min().item())
        assert 0 <= min_gid and max_gid < num_global_motifs, (
            f"motif_gidx out of bounds: min={min_gid}, max={max_gid}, "
            f"num_global_motifs={num_global_motifs}"
        )

        # Motif pooling（得到 2D 表征，hidden 维度）
        h_motif_hidden = self.motif_pool(x, motif_gidx, num_global_motifs)
        assert h_motif_hidden.shape == (num_global_motifs, self.hidden)

        # Teacher 3D Encoder（仅预训练阶段使用），返回 motif 级 teacher 表征投影到 z3d 维
        teacher_z = None
        if self.teacher_enabled and self.use_teacher_for_pt:
            assert hasattr(data, 'pos'), 'teacher.enabled=True 但 batch 中无 pos'
            # 教师网络对数值稳定性更敏感，强制在 FP32 通道前向，避免 AMP 半精度导致的溢出/NaN
            with torch.cuda.amp.autocast(enabled=False):
                z_i64 = data.z  # long 不受 autocast 影响
                pos_f32 = data.pos.float()
                batch_i64 = data.batch
                h_atom_3d = self.teacher(z_i64, pos_f32, batch_i64)
                # 一阶均值
                mu = self.motif_pool(h_atom_3d, motif_gidx, num_global_motifs)
                # 二阶：均方 - 均值平方，截断到非负，提供局部几何的二阶统计
                e2 = self.motif_pool(h_atom_3d * h_atom_3d, motif_gidx, num_global_motifs)
                var = torch.clamp(e2 - mu * mu, min=0.0)
                h_motif_local = torch.cat([mu, var], dim=-1)
                tz = self.teacher_proj(h_motif_local)
                # 将 teacher 输出对齐到与 z_gt 相同的标准化空间
                if self._z3d_norm_set:
                    tz = (tz - self.z3d_mean) / self.z3d_std
                teacher_z = tz
        # 目标（教师）分布：基于 teacher_z（若无则用 z_gt）与 code 的距离
        # 注意：数据中的 motif_target 已经在数据集构建时标准化到与 z_gt 一致的空间
        with torch.cuda.amp.autocast(enabled=False):
            if teacher_z is None:
                t = data.motif_target.float()
            else:
                t = teacher_z.float()
            # [N,K] = ||t||^2 + ||E||^2 - 2 t E^T
            t2 = (t * t).sum(dim=1, keepdim=True)
            c2 = (self.codebook.codes * self.codebook.codes).sum(dim=1).view(1, -1)
            d2_t = t2 + c2 - 2.0 * t.matmul(self.codebook.codes.t())
            p_t = F.softmax(-d2_t / max(self.teacher_temperature, 1e-8), dim=-1)
            # 记录基于教师的最近邻索引（用于监控）
            teacher_nn_index = torch.argmin(d2_t, dim=-1)

        # h_commit 仅用于 commit 损失（把 2D 表征投影到 z3d 以便与 code 对齐）
        h_commit = self.commit_proj(h_motif_hidden)

        # 可学习路由分布：线性 Router + Gumbel-Softmax
        logits = self.router(h_motif_hidden)
        assert logits.shape == (num_global_motifs, self.codebook_size)
        p_r = F.gumbel_softmax(logits, tau=max(self._tau_r, 1e-8), hard=False, dim=-1)

        # 融合（PoE）：y_soft ∝ p_r^beta * p_t^(1-beta)
        eps = 1e-9
        pr = torch.clamp(p_r, min=eps)
        pt = torch.clamp(p_t, min=eps)
        y_soft = pr

        # 可选 Top-K 稀疏化/硬化
        # 约定：
        #   - router_topk == 0: 禁用稀疏与硬化，直接使用软分布（期望向量），避免早期梯度饥饿
        #   - router_topk == 1: Top-1 ST 硬化
        #   - router_topk  > 1: Top-K 稀疏 + ST（按被选中的 K 个概率归一化）
        if self.router_topk == 0:
            topk_idx = torch.argmax(y_soft, dim=-1)  # 仅用于监控
            y = y_soft  # 不进行硬化，直接使用软分布
            y_hard = y_soft  # 记录到 router_info 供可视化
        elif self.router_topk > 1:
            k = min(self.router_topk, self.codebook_size)
            vals, idx = torch.topk(y_soft, k=k, dim=-1)
            mask = torch.zeros_like(y_soft)
            mask.scatter_(dim=1, index=idx, src=torch.ones_like(vals))
            y_sparse = y_soft * mask
            y_sparse = y_sparse / (y_sparse.sum(dim=-1, keepdim=True) + eps)
            # ST：前向稀疏、反向走原 y_soft
            y_hard = torch.zeros_like(y_soft)
            y_hard.scatter_(1, idx, vals / (vals.sum(dim=-1, keepdim=True) + eps))
            y = y_hard + (y_soft - y_hard).detach()
            topk_idx = idx[:, 0]
        else:
            topk_idx = torch.argmax(y_soft, dim=-1)
            y_hard = torch.zeros_like(y_soft)
            y_hard.scatter_(1, topk_idx.view(-1, 1), 1.0)
            y = y_hard + (y_soft - y_hard).detach()

        # 选取 code 的期望向量（ST 允许反传）
        e_k = y.matmul(self.codebook.codes)
        assert e_k.shape == (num_global_motifs, self.z3d_dim)

        # Expert 形变
        expert_in = torch.cat([h_motif_hidden, e_k], dim=-1)
        delta = self.expert(expert_in)
        assert delta.shape == (num_global_motifs, self.z3d_dim)

        z_hat = e_k + delta
        assert h_commit.shape == (num_global_motifs, self.z3d_dim)

        router_info = {
            'p_r': p_r,
            'p_t': p_t,
            'y_soft': y_soft,
            'y_hard': y_hard,
            'logits': logits,
            'teacher_nn_index': teacher_nn_index,
        }
        if self.router_topk > 1:
            router_info['topk_indices'] = idx
        if teacher_z is not None:
            router_info['teacher_z'] = teacher_z
        return z_hat, h_commit, e_k, logits, topk_idx, router_info

    # 允许训练过程中调整 Top-K 稀疏化强度
    # 约定：k==0 表示禁用稀疏与硬化（用软分布期望）；k==1 为 Top-1；k>1 为 Top-K
    def set_topk(self, k: int) -> None:
        self.router_topk = int(k)

    def readout(self, h_motif: torch.Tensor, motif_gidx: torch.Tensor, num_graphs: int) -> torch.Tensor:
        # 将 motif 特征聚合为分子级表示
        if self.readout_type == 'sum':
            # 这里 motif_gidx 是 motif 的全局索引；需要 motif->graph 的映射
            # 根据 data.batch 构造的 motif 偏移，motif 的 graph id 顺序与 num_motifs 向量一致
            # 简单起见，这里由外部提供 motif_graph_ids 更合理；本函数仅提供聚合原语
            raise RuntimeError('请使用外部提供的 motif->graph 映射执行 readout')
        else:
            raise RuntimeError('请使用外部提供的 motif->graph 映射执行 attention readout')
