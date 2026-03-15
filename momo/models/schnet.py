import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi as PI
from torch_geometric.nn import radius_graph


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=10.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class CFConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_filters, mlp, cutoff):
        super().__init__()
        # Use the torch.nn module (aliased as nn) for layers; avoid shadowing by args
        self.lin1 = nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = nn.Linear(num_filters, out_channels)
        self.mlp = mlp
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def message(self, x_j, W):
        return x_j * W

    def forward(self, x, edge_index, edge_weight, edge_attr):
        row, col = edge_index
        # 先通过线性层，随后以其输出 dtype 作为基准，避免 AMP 下 dtype 不一致
        x = self.lin1(x)
        dtype = x.dtype
        C = 0.5 * (torch.cos(edge_weight.to(dtype) * PI / self.cutoff) + 1.0)
        C = C.to(dtype)
        W = self.mlp(edge_attr.to(dtype))
        W = (W * C.view(-1, 1)).to(dtype)
        out = torch.zeros_like(x)
        src = self.message(x[col], W)
        out.index_add_(0, row, src)
        out = self.lin2(out)
        return out


class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            nn.Softplus(),
            nn.Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters, self.mlp, cutoff)
        self.act = nn.Softplus()
        self.lin = nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class AtomSchNet(nn.Module):
    """A minimal SchNet returning atom-level embeddings (no pooling)."""
    def __init__(self, hidden_channels=128, num_filters=128, num_interactions=6, num_gaussians=50, cutoff=10.0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.embedding = nn.Embedding(119, hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        self.cutoff = cutoff

        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians, num_filters, cutoff)
            self.interactions.append(block)

        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.act = nn.Softplus()
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        assert z.dim() == 1 and z.dtype == torch.long
        h = self.embedding(z)
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)
        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)
        return h
