from typing import Tuple, Dict, Optional, List

import torch
from torch import nn
import torch.nn.functional as F
from .schnet import AtomSchNet
from torch_geometric.nn import GINConv, GINEConv
from torch_scatter import scatter_mean, scatter_add, scatter_max


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        # Expose channel metadata so PyG (e.g., GINEConv) can
        # infer input/output channels from custom MLP wrappers.
        self.in_channels = in_dim
        self.out_channels = out_dim
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


class GINEStack(nn.Module):
    def __init__(self, hidden: int, num_layers: int, mlp_hidden: int, dropout: float, edge_dim: int):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            mlp = MLP(hidden, mlp_hidden, hidden, dropout)
            conv = GINEConv(mlp, edge_dim=edge_dim)
            layers.append(conv)
        self.layers = nn.ModuleList(layers)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        for conv in self.layers:
            x = conv(x, edge_index, edge_attr)
            x = self.act(x)
        return x


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
        # 边级目标维度（标准化后），默认 7
        self.edge_target_dim = int(mcfg.get('edge_target_dim', 7))
        assert self.edge_target_dim > 0, 'model.edge_target_dim 必须在配置中显式给出且 > 0'
        self.codebook_size = int(mcfg['codebook_size'])
        self.num_experts = int(mcfg.get('num_experts', 8))
        assert self.num_experts >= 2
        # Δ 缩放系数（限制容量）
        self.delta_scale = float(mcfg.get('delta_scale', 0.2))
        self.ctx_delta_scale = float(mcfg.get('ctx_delta_scale', 0.1))
        self.ctx_delta_bottleneck = int(mcfg.get('ctx_delta_bottleneck', max(32, self.hidden // 4)))
        assert self.ctx_delta_bottleneck > 0

        self.atom_encoder = AtomEncoder(int(dcfg['max_atomic_num']), self.hidden)
        # 双分支 GIN：local（仅 motif 内边）与 ctx（仅跨 motif 边）
        self.gin_local = GINStack(self.hidden, int(mcfg['num_gin_layers']), int(mcfg['gin_mlp_hidden']), float(mcfg['dropout']))
        # 不再使用原子级跨 motif 边 ctx；上下文语义仅通过 motif 图传递
        # motif 边特征维度（预处理生成：4 bond onehot + 2 aromatic + 2 rank_norm）
        self.edge_attr_dim = int(mcfg.get('motif_edge_dim', 8))
        # 邻居 motif 类型 embedding（用于边语义增强）
        self.neighbor_type_vocab = int(mcfg.get('neighbor_type_vocab_size', 100000))
        self.neighbor_type_emb_dim = int(mcfg.get('neighbor_type_emb_dim', 16))
        self.neighbor_type_emb = nn.Embedding(self.neighbor_type_vocab, self.neighbor_type_emb_dim)
        # 附件位置 embedding（motif 内局部锚点索引）
        self.attach_pos_vocab = int(mcfg.get('attach_pos_vocab_size', 64))
        self.attach_pos_emb_dim = int(mcfg.get('attach_pos_emb_dim', 8))
        self.attach_pos_emb = nn.Embedding(self.attach_pos_vocab, self.attach_pos_emb_dim)
        # motif 级上下文 GNN（将邻居 motif 语义汇聚到当前 motif，带边语义）
        self.gin_motif_ctx = GINEStack(
            self.hidden,
            int(mcfg['num_gin_layers']),
            int(mcfg['gin_mlp_hidden']),
            float(mcfg['dropout']),
            edge_dim=(self.edge_attr_dim + self.neighbor_type_emb_dim + 2 * self.attach_pos_emb_dim),
        )
        self.vq_proj = nn.Sequential(
            nn.LayerNorm(self.hidden),
            nn.Linear(self.hidden, self.hidden),
        )
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

        # 路由控制：仅用于距离分布的软诊断，不改变最近邻硬量化。
        self.router_tau: float = float(rcfg.get('gumbel_tau_start', 1.0))
        # 主 3D 监督头：用增强后的 motif 表示预测 teacher motif latent。
        self.main_3d_head = nn.Sequential(
            nn.Linear(self.hidden, int(mcfg['gin_mlp_hidden'])),
            nn.ReLU(),
            nn.Dropout(float(mcfg['dropout'])),
            nn.Linear(int(mcfg['gin_mlp_hidden']), self.hidden),
        )
        # 边级 Δ -> 几何目标（7 维）
        self.latent_to_edge = nn.Linear(self.hidden, self.edge_target_dim)

        # Teacher (3D encoder) 配置
        tcfg = cfg.get('teacher', None)
        self.teacher_enabled = bool(tcfg and tcfg.get('enabled', False))
        # 仅在 teacher_enabled=True 时计算 p_t；不提供“兜底”路径

        readout = str(mcfg['readout']).lower()
        assert readout in ['sum', 'attention']
        self.readout_type = readout
        if readout == 'attention':
            self.att_proj = nn.Linear(self.hidden + self.hidden, 1)

        # 邻居定向 Δ：基于受约束的 2D 上下文（仅邻居类型、键类型与附件位置）按边生成残差并聚合
        # 输入严格限制为 [e_hat[src], edge_ctx_uv]，避免以 h_ctx 作为强通道直接复原几何
        ed_in = (self.hidden) + (self.edge_attr_dim + self.neighbor_type_emb_dim + 2 * self.attach_pos_emb_dim)
        mid = int(mcfg.get('expert_mlp_hidden', self.hidden))
        self.edge_delta_mlp = nn.Sequential(
            nn.Linear(ed_in, mid), nn.ReLU(), nn.Linear(mid, self.hidden)
        )
        self.edge_gate_mlp = nn.Sequential(
            nn.Linear(ed_in, mid), nn.ReLU(), nn.Linear(mid, 1)
        )
        # 弱上下文补充分支：先从 h_ctx 中减掉与 e_hat 平行的分量，再通过小瓶颈预测剩余残差。
        self.ctx_query = nn.Linear(self.hidden, self.ctx_delta_bottleneck)
        self.ctx_code = nn.Linear(self.hidden, self.ctx_delta_bottleneck)
        self.ctx_delta_mlp = nn.Sequential(
            nn.LayerNorm(self.ctx_delta_bottleneck),
            nn.Linear(self.ctx_delta_bottleneck, mid),
            nn.ReLU(),
            nn.Linear(mid, self.hidden),
        )
        # ctx 由 unique(ctx, local) 与边语义两路共同决定，使用可学习门控做逐通道融合。
        self.ctx_fuse_gate = nn.Sequential(
            nn.LayerNorm(2 * self.hidden),
            nn.Linear(2 * self.hidden, mid),
            nn.ReLU(),
            nn.Linear(mid, self.hidden),
        )
        self.ctx_dropout = nn.Dropout(float(mcfg['dropout']))

        # 教师温度
        self.teacher_temperature = float(rcfg.get('teacher_temperature', 1.5))

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

    def forward(self, data: 'torch_geometric.data.Batch', motif_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # 原子编码
        x0 = self.atom_encoder(data.z)
        assert x0.dim() == 2 and x0.size(1) == self.hidden

        # 构造 motif 内/跨 motif 的边索引（严格区分职责，不做差分近似）
        ei = data.edge_index
        same = (data.motif_id[ei[0]] == data.motif_id[ei[1]])
        edge_index_local = ei[:, same]

        # 原子级编码（仅 motif 内边）：得到 motif 本体的原子表征
        x_local = self.gin_local(x0, edge_index_local)

        # 全局 motif 索引
        from momo.data.pcqm4mv2 import motif_global_index, _infer_num_motifs_per_graph
        motif_gidx = motif_global_index(data.motif_id, data.batch)
        # 基于 batch 内部一致性推断每图 motif 数，避免 DataLoader 拼接歧义
        inferred_counts = _infer_num_motifs_per_graph(data.motif_id, data.batch)
        num_global_motifs = int(inferred_counts.sum().item())
        # 在批次内，确保 motif_gidx 索引未越界
        max_gid = int(motif_gidx.max().item())
        min_gid = int(motif_gidx.min().item())
        assert 0 <= min_gid and max_gid < num_global_motifs, (
            f"motif_gidx out of bounds: min={min_gid}, max={max_gid}, "
            f"num_global_motifs={num_global_motifs}"
        )

        # Motif pooling：得到无上下文本体表征
        h_motif_local = self.motif_pool(x_local, motif_gidx, num_global_motifs)
        # MMM：先在 motif 级对本体表征应用 mask，再计算上下文，避免信息泄漏
        if motif_mask is not None:
            assert motif_mask.dtype == torch.bool
            assert motif_mask.dim() == 1 and motif_mask.numel() == h_motif_local.size(0)
            token = self.mask_token.to(dtype=h_motif_local.dtype, device=h_motif_local.device)
            h_motif_local = torch.where(
                motif_mask.view(-1, 1),
                token.view(1, -1).expand_as(h_motif_local),
                h_motif_local,
            )
        # 构造 motif 级图（批内全局 motif 索引空间），并在 motif 图上做消息传递得到上下文语义
        assert hasattr(data, 'motif_edge_index'), '预处理必须提供 motif_edge_index'
        motif_edge_index = data.motif_edge_index
        # 在 motif 级图上传播，得到“邻居 motif 语义”（带边语义 + 邻居类型嵌入）
        # 邻居类型与附件位置嵌入（逐边提供 src/dst 的局部索引；作为受约束的 2D 上下文）
        assert hasattr(data, 'motif_edge_neighbor_type'), '预处理必须提供 motif_edge_neighbor_type'
        assert hasattr(data, 'motif_edge_attach_pos_src'), '预处理必须提供 motif_edge_attach_pos_src'
        assert hasattr(data, 'motif_edge_attach_pos_dst'), '预处理必须提供 motif_edge_attach_pos_dst'
        nb_type = data.motif_edge_neighbor_type.long()
        nb_emb = self.neighbor_type_emb(nb_type)
        # 逐边附件位置（预处理直接提供）
        att_src = data.motif_edge_attach_pos_src.long()
        att_dst = data.motif_edge_attach_pos_dst.long()
        if att_src.numel() > 0:
            assert int(att_src.min().item()) >= 0 and int(att_src.max().item()) < self.attach_pos_vocab
            assert int(att_dst.min().item()) >= 0 and int(att_dst.max().item()) < self.attach_pos_vocab
        att_emb_src = self.attach_pos_emb(att_src)
        att_emb_dst = self.attach_pos_emb(att_dst)
        edge_attr_cat = torch.cat([data.motif_edge_attr, nb_emb, att_emb_src, att_emb_dst], dim=-1)

        assert motif_edge_index.dim() == 2 and motif_edge_index.size(0) == 2
        # if motif_edge_index.numel() > 0:
        #     mx = int(motif_edge_index.max().item())
        #     mn = int(motif_edge_index.min().item())
        #     assert 0 <= mn and mx < num_global_motifs, \
        #         f"batched motif_edge_index out of bounds: min={mn}, max={mx}, num_global_motifs={num_global_motifs}"

        h_motif_ctx = self.gin_motif_ctx(h_motif_local, motif_edge_index, edge_attr_cat)
        assert h_motif_local.shape == (num_global_motifs, self.hidden)
        assert h_motif_ctx.shape == (num_global_motifs, self.hidden)

        # Teacher（仅在预训练启用），输出到 D 维
        teacher_z = None
        template_z = None
        if self.teacher_enabled:
            assert hasattr(data, 'pos'), 'teacher.enabled=True 但 batch 中无 pos'
            assert hasattr(data, 'atom_local_coord'), 'template supervision requires atom_local_coord in batch'
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
                # 模板教师：只看 motif 内局部坐标，并按 motif_gidx 将原子聚合回 motif。
                local_pos_f32 = data.atom_local_coord.float()
                h_atom_template = self.teacher(z_i64, local_pos_f32, motif_gidx)
                mu_template = self.motif_pool(h_atom_template, motif_gidx, num_global_motifs)
                e2_template = self.motif_pool(h_atom_template * h_atom_template, motif_gidx, num_global_motifs)
                var_template = torch.clamp(e2_template - mu_template * mu_template, min=0.0)
                h_motif_template = torch.cat([mu_template, var_template], dim=-1)
                template_z = self.teacher_proj(h_motif_template)

        # 标准 VQ：连续 encoder latent 经过最近邻量化进入 codebook。
        z_e = self.vq_proj(h_motif_local)
        codes = self.codebook.codes
        with torch.cuda.amp.autocast(enabled=False):
            z_e_f = z_e.float()
            codes_f = codes.float()
            z2 = (z_e_f * z_e_f).sum(dim=-1, keepdim=True)
            c2 = (codes_f * codes_f).sum(dim=-1).view(1, -1)
            dist_sq = z2 + c2 - 2.0 * z_e_f.matmul(codes_f.t())
            dist_sq = torch.clamp(dist_sq, min=0.0)
        logits_code = -dist_sq
        p_code_soft = F.softmax(logits_code / max(self.router_tau, 1e-8), dim=-1)
        code_idx = torch.argmin(dist_sq, dim=-1)
        p_code = F.one_hot(code_idx, num_classes=self.codebook_size).to(z_e.dtype)
        z_q = self.codebook.lookup(code_idx)
        # Straight-through quantization：前向用离散 code，反向沿 encoder latent 传梯度。
        e_hat = z_e + (z_q - z_e).detach()

        # Expert 路由仅用于负载约束（可选）；Δ 由邻居定向边聚合产生
        logits_exp = self.expert_router(h_motif_ctx)
        p_exp = F.softmax(logits_exp, dim=-1)

        # 邻居定向 Δ：对每条 motif 边（u->v）计算边级残差并对同一 u 聚合（softmax 权重）。
        # 这条支路只读取 stop-grad 的离散模板，避免 edge 监督直接改写 VQ 主链。
        eidx = motif_edge_index  # [2, E]
        if eidx.numel() > 0:
            src = eidx[0]
            # 边语义直接参与训练；仅模板侧保留 stop-grad，避免 edge 监督直接改写 VQ 主链。
            e_attr = self.ctx_dropout(edge_attr_cat)
            proto_src = z_q.detach().index_select(0, src)
            edge_in = torch.cat([proto_src, e_attr], dim=-1)
            edge_delta = self.edge_delta_mlp(edge_in)  # [E,D]
            gate_logits = self.edge_gate_mlp(edge_in).squeeze(-1)  # [E]
            # softmax over edges group by src
            max_per_src, _ = scatter_max(gate_logits, src, dim=0, dim_size=num_global_motifs)
            max_g = max_per_src.index_select(0, src)
            y = torch.exp(gate_logits - max_g)
            sum_per_src = scatter_add(y, src, dim=0, dim_size=num_global_motifs)
            denom = sum_per_src.index_select(0, src) + 1e-12
            alpha = (y / denom).unsqueeze(-1)  # [E,1]
            # 监督和最终注入的 latent 使用同一缩放，避免 edge 支路目标与主链注入量不一致。
            edge_delta_scaled = self.delta_scale * edge_delta
            delta_edge = scatter_add(alpha * edge_delta_scaled, src, dim=0, dim_size=num_global_motifs)
            edge_delta_geom = self.latent_to_edge(edge_delta_scaled)
        else:
            delta_edge = torch.zeros_like(h_motif_local)
            edge_delta_geom = torch.zeros((0, self.edge_target_dim), device=h_motif_local.device, dtype=h_motif_local.dtype)

        # 上下文补充分支：从 h_ctx 中剥离与 local/codebook 平行的成分，得到 unique(ctx, local)。
        # h_ctx 直接参与训练；仅模板基底保留 stop-grad，确保 ctx 只补偏移、不反向拖动 local 原型。
        ctx_q = self.ctx_query(self.ctx_dropout(h_motif_ctx))
        code_basis = self.ctx_code(e_hat.detach())
        denom = (code_basis * code_basis).sum(dim=-1, keepdim=True) + 1e-12
        coeff = (ctx_q * code_basis).sum(dim=-1, keepdim=True) / denom
        ctx_parallel = coeff * code_basis
        ctx_unique = ctx_q - ctx_parallel
        delta_ctx = self.ctx_delta_mlp(ctx_unique)

        # 最终 ctx 由 unique(ctx, local) 与边语义两路偏移经可学习门控融合得到。
        delta_ctx_scaled = self.ctx_delta_scale * delta_ctx
        ctx_fuse_in = torch.cat([delta_ctx_scaled, delta_edge], dim=-1)
        ctx_fuse_weight = torch.sigmoid(self.ctx_fuse_gate(ctx_fuse_in))
        delta_scaled = ctx_fuse_weight * delta_ctx_scaled + (1.0 - ctx_fuse_weight) * delta_edge
        h_motif_enh = e_hat + delta_scaled

        # 边级监督目标（标准化后）
        edge_target = data.motif_edge_target.float()

        delta_out = delta_scaled
        delta_edge_out = delta_edge
        delta_ctx_out = delta_ctx_scaled

        router_info: Dict[str, torch.Tensor] = {
            'p_code': p_code,
            'p_code_soft': p_code_soft,
            'logits_code': logits_code,
            'p_exp': p_exp,
            'logits_exp': logits_exp,
            'codebook_codes': self.codebook.codes,
            'code_idx': code_idx,
            'z_e': z_e,
            'z_q': z_q,
            'e_hat': e_hat,
            'h_motif_enh': h_motif_enh,
            'h_motif_local': h_motif_local,
            'h_motif_ctx': h_motif_ctx,
            'ctx_unique': ctx_unique,
            'ctx_fuse_weight': ctx_fuse_weight,
            # 边级监督
            'edge_target': edge_target,
            'edge_delta_geom': edge_delta_geom,
        }
        router_info['delta'] = delta_out
        router_info['delta_edge'] = delta_edge_out
        router_info['delta_ctx'] = delta_ctx_out
        if teacher_z is not None:
            router_info['z3d_target'] = teacher_z
            router_info['z3d_pred'] = self.main_3d_head(h_motif_enh)
        if template_z is not None:
            router_info['z_template_target'] = template_z

        # 兼容返回形状：首个返回值用于 batch 大小统计，直接返回 h_motif_local
        return h_motif_local, h_motif_enh, router_info

    def set_router_temperature(self, tau: float) -> None:
        # 仅影响按距离得到的软分布诊断，不改变最近邻硬量化。
        self.router_tau = float(tau)

    def readout(self, h_motif: torch.Tensor, motif_gidx: torch.Tensor, num_graphs: int) -> torch.Tensor:
        # 将 motif 特征聚合为分子级表示
        if self.readout_type == 'sum':
            # 这里 motif_gidx 是 motif 的全局索引；需要 motif->graph 的映射
            # 根据 data.batch 构造的 motif 偏移，motif 的 graph id 顺序与 num_motifs 向量一致
            # 简单起见，这里由外部提供 motif_graph_ids 更合理；本函数仅提供聚合原语
            raise RuntimeError('请使用外部提供的 motif->graph 映射执行 readout')
        else:
            raise RuntimeError('请使用外部提供的 motif->graph 映射执行 attention readout')
