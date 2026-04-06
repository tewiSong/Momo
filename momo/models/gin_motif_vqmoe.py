from typing import Tuple, Dict, Optional, List

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


class PrototypeRouter(nn.Module):
    def __init__(self, in_dim: int, codebook_size: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, codebook_size)

    def forward(self, h_motif: torch.Tensor) -> torch.Tensor:
        logits = self.fc(h_motif)
        return logits


class Codebook(nn.Module):
    def __init__(self, codebook_size: int, code_dim: int):
        super().__init__()
        self.codes = nn.Parameter(torch.randn(codebook_size, code_dim) * 0.02)

    def lookup(self, idx: torch.Tensor) -> torch.Tensor:
        return self.codes[idx]


class ExpertRouter(nn.Module):
    def __init__(self, in_dim: int, num_experts: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_experts)

    def forward(self, h_motif: torch.Tensor) -> torch.Tensor:
        logits = self.fc(h_motif)
        return logits


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
        rcfg = cfg.get('router', None)
        assert rcfg is not None, "router 配置缺失"
        self.hidden = int(mcfg['hidden_dim'])
        # 显式几何标签维度（标准化后的 z3d^GT 维度）
        self.geom_dim = int(mcfg.get('geom_dim', -1))
        assert self.geom_dim > 0, 'model.geom_dim 必须在配置中显式给出且 > 0'
        self.codebook_size = int(mcfg['codebook_size'])
        # Expert 路由参数
        self.num_experts = int(mcfg.get('num_experts', 8))
        assert self.num_experts >= 2
        self.expert_topk = int(rcfg.get('expert_topk', 2))
        assert 1 <= self.expert_topk <= self.num_experts
        # Prototype 路由 Top‑K（用于 ST 硬选择）
        self.proto_topk = int(rcfg.get('topk', 1))
        assert self.proto_topk >= 1
        # Δ 缩放系数（限制容量）
        self.delta_scale = float(mcfg.get('delta_scale', 0.2))

        self.atom_encoder = AtomEncoder(int(dcfg['max_atomic_num']), self.hidden)
        # 双分支 GIN：local（仅 motif 内边）与 ctx（仅跨 motif 边）
        self.gin_local = GINStack(self.hidden, int(mcfg['num_gin_layers']), int(mcfg['gin_mlp_hidden']), float(mcfg['dropout']))
        self.gin_ctx = GINStack(self.hidden, int(mcfg['num_gin_layers']), int(mcfg['gin_mlp_hidden']), float(mcfg['dropout']))
        self.proto_router = PrototypeRouter(self.hidden, self.codebook_size)
        # Codebook 维度与 2D latent 统一为 D=hidden
        self.codebook = Codebook(self.codebook_size, self.hidden)
        # Expert 路由与多专家
        self.expert_router = ExpertRouter(self.hidden, self.num_experts)
        self.experts = nn.ModuleList([
            ExpertMLP(self.hidden + self.hidden, int(mcfg['expert_mlp_hidden']), self.hidden, float(mcfg['dropout']))
            for _ in range(self.num_experts)
        ])
        # MMM 掩码 token（用于替换被 mask 的 motif 表征）
        self.mask_token = nn.Parameter(torch.zeros(self.hidden))

        # 显式几何标签 <-> 主 latent 的双向桥接头
        # zgt_to_latent: 将 7 维几何（或 cfg.geom_dim）投影到 D 维 latent 空间，用于原型监督（p_gt 与最近原型）与 VQ
        # latent_to_zgt: 将 D 维 latent 回归到几何标签，用于主重建损失
        self.zgt_to_latent = nn.Sequential(
            nn.LayerNorm(self.geom_dim),
            nn.Linear(self.geom_dim, self.hidden),
        )
        self.latent_to_zgt = nn.Linear(self.hidden, self.geom_dim)

        # Teacher (3D encoder) 配置
        tcfg = cfg.get('teacher', None)
        self.teacher_enabled = bool(tcfg and tcfg.get('enabled', False))
        # 仅在 teacher_enabled=True 时计算 p_t；不提供“兜底”路径

        readout = str(mcfg['readout']).lower()
        assert readout in ['sum', 'attention']
        self.readout_type = readout
        if readout == 'attention':
            self.att_proj = nn.Linear(self.hidden + self.hidden, 1)

        # 教师/几何分布温度
        self.teacher_temperature = float(rcfg.get('teacher_temperature', 1.5))
        self.gt_temperature = float(rcfg.get('gt_temperature', rcfg.get('distance_temperature', 1.0)))

        # Teacher 模型与投影
        if self.teacher_enabled:
            th = int(tcfg['hidden_dim'])
            self.teacher = AtomSchNet(
                hidden_channels=th,
                num_filters=int(tcfg['num_filters']),
                num_interactions=int(tcfg['num_interactions']),
                num_gaussians=int(tcfg['num_gaussians']),
                cutoff=float(tcfg['cutoff']),
                max_num_neighbors=int(tcfg.get('max_num_neighbors', 32)),
                normalize_messages=bool(tcfg.get('normalize_messages', True)),
                activation=str(tcfg.get('activation', 'shifted_softplus')),
            )
            # 高阶聚合映射到 D 维（与 latent 统一）
            in_dim = th * 2
            self.teacher_proj = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, th),
                nn.ReLU(),
                nn.Linear(th, self.hidden),
            )

    def set_teacher_temperature(self, t: float) -> None:
        self.teacher_temperature = float(t)

    @staticmethod
    def motif_pool(x_atom: torch.Tensor, motif_global_idx: torch.Tensor, num_global_motifs: int) -> torch.Tensor:
        assert motif_global_idx.dim() == 1 and x_atom.size(0) == motif_global_idx.size(0)
        h_motif = scatter_mean(x_atom, motif_global_idx, dim=0, dim_size=num_global_motifs)
        assert h_motif.shape[0] == num_global_motifs
        return h_motif

    def forward(self, data: 'torch_geometric.data.Batch', motif_mask: Optional[torch.Tensor] = None, *, freeze_delta: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # 原子编码
        x0 = self.atom_encoder(data.z)
        assert x0.dim() == 2 and x0.size(1) == self.hidden

        # 构造 motif 内/跨 motif 的边索引（严格区分职责，不做差分近似）
        ei = data.edge_index
        same = (data.motif_id[ei[0]] == data.motif_id[ei[1]])
        edge_index_local = ei[:, same]
        edge_index_ctx = ei[:, ~same]

        # 两路 GIN 编码
        x_local = self.gin_local(x0, edge_index_local)
        x_ctx = self.gin_ctx(x0, edge_index_ctx)

        # 全局 motif 索引
        from momo.data.pcqm4mv2 import motif_global_index, _infer_num_motifs_per_graph
        motif_gidx = motif_global_index(data.motif_id, data.batch)
        # 基于 batch 内部一致性推断每图 motif 数，避免 DataLoader 拼接歧义
        inferred_counts = _infer_num_motifs_per_graph(data.motif_id, data.batch)
        num_global_motifs = int(inferred_counts.sum().item())
        # motif_target（显式几何标签）在预训练是必须的
        assert hasattr(data, 'motif_target') and int(data.motif_target.size(0)) == num_global_motifs
        assert int(data.motif_target.size(1)) == self.geom_dim, (
            f"geom_dim mismatch: data={int(data.motif_target.size(1))} vs cfg.model.geom_dim={self.geom_dim}"
        )
        # 在批次内，确保 motif_gidx 索引未越界
        max_gid = int(motif_gidx.max().item())
        min_gid = int(motif_gidx.min().item())
        assert 0 <= min_gid and max_gid < num_global_motifs, (
            f"motif_gidx out of bounds: min={min_gid}, max={max_gid}, "
            f"num_global_motifs={num_global_motifs}"
        )

        # Motif pooling：得到无上下文/上下文两路表征
        h_motif_local = self.motif_pool(x_local, motif_gidx, num_global_motifs)
        h_motif_ctx = self.motif_pool(x_ctx, motif_gidx, num_global_motifs)
        assert h_motif_local.shape == (num_global_motifs, self.hidden)
        assert h_motif_ctx.shape == (num_global_motifs, self.hidden)

        # Teacher（仅在预训练启用），输出到 D 维
        teacher_z = None
        if self.teacher_enabled:
            assert hasattr(data, 'pos'), 'teacher.enabled=True 但 batch 中无 pos'
            with torch.cuda.amp.autocast(enabled=False):
                z_i64 = data.z  # long 不受 autocast 影响
                pos_f32 = data.pos.float()
                batch_i64 = data.batch
                h_atom_3d = self.teacher(z_i64, pos_f32, batch_i64)
                mu = self.motif_pool(h_atom_3d, motif_gidx, num_global_motifs)
                e2 = self.motif_pool(h_atom_3d * h_atom_3d, motif_gidx, num_global_motifs)
                var = torch.clamp(e2 - mu * mu, min=0.0)
                h_motif_local_t = torch.cat([mu, var], dim=-1)
                tz = self.teacher_proj(h_motif_local_t)
                teacher_z = tz

        # MMM：将被 mask 的 motifs 的 2D 表征替换为 mask_token（非就地，显式对齐 dtype/device）
        if motif_mask is not None:
            assert motif_mask.dtype == torch.bool
            assert motif_mask.dim() == 1 and motif_mask.numel() == h_motif_local.size(0)
            token = self.mask_token.to(dtype=h_motif_local.dtype, device=h_motif_local.device)
            h_motif_local = torch.where(
                motif_mask.view(-1, 1),
                token.view(1, -1).expand_as(h_motif_local),
                h_motif_local,
            )

        # Prototype 路由（仅看 2D latent）
        # 原型路由仅看无上下文的 h_motif_local
        logits_code = self.proto_router(h_motif_local)
        p_code_soft = F.softmax(logits_code, dim=-1)
        # Top‑K 选择（原型侧）：前向仅由被选中的 code 组成，反向经由软分布
        k_proto = min(self.proto_topk, self.codebook_size)
        assert k_proto >= 1, 'proto_topk must be >= 1'
        vals, idx = torch.topk(p_code_soft, k=k_proto, dim=-1)
        y_hard = torch.zeros_like(p_code_soft)
        if k_proto == 1:
            y_hard.scatter_(1, idx, 1.0)
        else:
            # 仅在被选中的 top‑k 内按相对强度加权，保证 e_hat 不退化为全局均值
            y_hard.scatter_(1, idx, vals / vals.sum(dim=-1, keepdim=True))
        # Straight‑Through: 前向= y_hard，反向= y_soft
        y_code = p_code_soft + (y_hard - p_code_soft).detach()
        # 原型记忆信号（ST）
        e_hat = y_code @ self.codebook.codes  # [N,D]

        # Expert 路由与稀疏选通
        # 专家路由仅看上下文表征
        logits_exp = self.expert_router(h_motif_ctx)
        p_exp = F.softmax(logits_exp, dim=-1)
        k = self.expert_topk
        exp_vals, exp_idx = torch.topk(p_exp, k=k, dim=-1)
        exp_vals_norm = exp_vals / exp_vals.sum(dim=-1, keepdim=True)
        # 计算每个选中 expert 的输出并加权
        expert_inputs = torch.cat([h_motif_ctx, e_hat], dim=-1)
        deltas: List[torch.Tensor] = []
        for j in range(k):
            ej_idx = exp_idx[:, j]
            # 收集对应 expert 的输出
            parts = []
            for i in range(self.num_experts):
                sel = (ej_idx == i)
                if sel.any():
                    out_i = self.experts[i](expert_inputs[sel])
                    parts.append((sel, out_i))
            # 重组为完整顺序张量
            delta_j = torch.zeros_like(h_motif_ctx)
            for sel, val in parts:
                delta_j[sel] = val
            deltas.append(delta_j)
        # 加权求和（得到 Δ_raw）
        delta = torch.zeros_like(h_motif_ctx)
        for j in range(k):
            wj = exp_vals_norm[:, j].view(-1, 1)
            delta = delta + wj * deltas[j]

        # MoE 增强后的表示（以 e_hat 为主，Δ 受缩放）
        delta_scaled = self.delta_scale * delta
        h_motif_enh = e_hat + delta_scaled

        # 几何 GT 及其投影到 latent
        # 定义“模板目标”：从几何 GT 的 latent 中减去上下文修正（Δ）在 latent 空间的贡献
        # 冻结期模板目标不应受 Δ 分支影响
        z3d_gt = data.motif_target.float()
        zgt_proj = self.zgt_to_latent(z3d_gt)
        with torch.cuda.amp.autocast(enabled=False):
            delta_for_tpl = torch.zeros_like(delta_scaled) if freeze_delta else delta_scaled.detach()
            z_template = zgt_proj.float() - delta_for_tpl.float()
            z2 = (z_template * z_template).sum(dim=1, keepdim=True)
            c2 = (self.codebook.codes * self.codebook.codes).sum(dim=1).view(1, -1)
            d2_tpl = z2 + c2 - 2.0 * z_template.matmul(self.codebook.codes.t())
            p_gt = F.softmax(-d2_tpl / max(self.gt_temperature, 1e-8), dim=-1)
            gt_nn_index = torch.argmin(d2_tpl, dim=-1)

        # 几何重建（从增强后 latent 回归回几何标签空间）与原型解码（仅用于监控）
        z_pred = self.latent_to_zgt(h_motif_enh)
        z_proto = self.latent_to_zgt(e_hat)

        # 冻结 Δ：仅让原型生效（影响所有调用者，包括主链路与辅助分支）
        if freeze_delta:
            delta_out = torch.zeros_like(e_hat)
            h_motif_enh = e_hat
            z_pred = z_proto
        else:
            delta_out = delta_scaled

        # 教师分布 p_t（仅在启用教师时计算）：仅作为可选参考，不作为主监督
        p_t = None
        teacher_nn_index = None
        if teacher_z is not None:
            with torch.cuda.amp.autocast(enabled=False):
                t = teacher_z.float()
                t2 = (t * t).sum(dim=1, keepdim=True)
                c2 = (self.codebook.codes * self.codebook.codes).sum(dim=1).view(1, -1)
                d2_t = t2 + c2 - 2.0 * t.matmul(self.codebook.codes.t())
                p_t = F.softmax(-d2_t / self.teacher_temperature, dim=-1)
                teacher_nn_index = torch.argmin(d2_t, dim=-1)

        router_info: Dict[str, torch.Tensor] = {
            'p_code': y_code,
            'p_code_soft': p_code_soft,
            'logits_code': logits_code,
            'p_exp': p_exp,
            'logits_exp': logits_exp,
            'exp_topk_idx': exp_idx,
            'codebook_codes': self.codebook.codes,
            'e_hat': e_hat,
            'h_motif_enh': h_motif_enh,
            'h_motif_local': h_motif_local,
            'h_motif_ctx': h_motif_ctx,
            # 显式几何标签与其投影
            'z3d_gt': z3d_gt,
            'zgt_proj': zgt_proj,
            'z_template': z_template,
            'p_gt': p_gt,
            'gt_nn_index': gt_nn_index,
            'z_pred': z_pred,
            'z_proto': z_proto,
        }
        if p_t is not None:
            router_info['aux_teacher_p'] = p_t
        if teacher_nn_index is not None:
            router_info['aux_teacher_nn_index'] = teacher_nn_index
        if teacher_z is not None:
            router_info['aux_teacher_z'] = teacher_z
        # 兼容旧日志键（使用 soft 分布作为 y_soft）
        router_info['y_soft'] = p_code_soft
        # 导出原型与专家的选通信息与残差
        router_info['proto_topk_idx'] = idx
        router_info['delta'] = delta_out
        router_info['e_hat'] = e_hat

        # 兼容返回形状：首个返回值用于 batch 大小统计，直接返回 h_motif_local
        return h_motif_local, h_motif_enh, router_info

    def set_topk(self, k: int) -> None:
        # 仅影响专家路由的 top-k
        self.expert_topk = int(k)

    def set_proto_topk(self, k: int) -> None:
        self.proto_topk = int(k)

    def readout(self, h_motif: torch.Tensor, motif_gidx: torch.Tensor, num_graphs: int) -> torch.Tensor:
        # 将 motif 特征聚合为分子级表示
        if self.readout_type == 'sum':
            # 这里 motif_gidx 是 motif 的全局索引；需要 motif->graph 的映射
            # 根据 data.batch 构造的 motif 偏移，motif 的 graph id 顺序与 num_motifs 向量一致
            # 简单起见，这里由外部提供 motif_graph_ids 更合理；本函数仅提供聚合原语
            raise RuntimeError('请使用外部提供的 motif->graph 映射执行 readout')
        else:
            raise RuntimeError('请使用外部提供的 motif->graph 映射执行 attention readout')
