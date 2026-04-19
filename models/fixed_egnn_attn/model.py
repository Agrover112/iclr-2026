"""
FixedEGNNAttn — FixedEGNN with multi-head GAT-style softmax attention.

Evolution from FixedEGNNGated:

  Gated:     g_ij = sigmoid(Linear(m_ij))                (unnormalized, per edge)
             agg_i = Σ_j g_ij · m_ij

  Attn:      e_ij^(h) = Linear_h(m_ij) for each head h in 1..H
             α_ij^(h) = softmax_{j: dst[j]=i} e_ij^(h)   (normalized per receiver × head)
             agg_i = concat_h [ Σ_j α_ij^(h) · m_ij^(h) ]

The message m_ij (hidden dim) is split into H slices of size hidden/H. Each
head softmax-normalizes its own per-edge logit over the receiver's neighbors
and aggregates its own slice. Slices are concatenated back to hidden dim.

Why multi-head: the per-edge message already carries distinct physical signals
(wall-normal via ∇UDF in h, streamwise via vel_proj, proximity via dist², shear
via h[src]−h[dst]). Independent heads with their own softmax let the network
specialize different heads to different signals via gradient descent — same
inductive bias that makes multi-head attention work in transformers/GAT.

Param cost: one Linear(hidden, H) per layer. For d=6 h=32 H=4 that's 132·6
= 792 extra params. Negligible.

Zero-init: all logits = 0 → softmax = 1/k uniform per head → mean aggregation
at init, same neutral-start philosophy as the parent class.
"""

import torch
import torch.nn as nn
from torch import Tensor

try:
    from torch_scatter import scatter_add, scatter_max, scatter_mean
except ImportError:  # pragma: no cover — CPU-only machines without torch_scatter
    scatter_add = scatter_max = scatter_mean = None

from models.fixed_egnn.model import FixedEGNNModel, FixedEGNNLayer


class FixedEGNNAttnLayer(FixedEGNNLayer):
    """EGNN layer with multi-head softmax-over-neighbors attention aggregation."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0,
                 update_coords: bool = False, heads: int = 4):
        super().__init__(hidden_dim, dropout=dropout, update_coords=update_coords)
        if hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by heads={heads}"
            )
        self.heads = heads
        self.head_dim = hidden_dim // heads
        # One logit per edge per head; read from the full message so each head
        # can attend on any feature, but aggregates only its own slice.
        self.attn = nn.Linear(hidden_dim, heads)
        nn.init.zeros_(self.attn.weight)
        nn.init.zeros_(self.attn.bias)

    def forward(
        self, h: Tensor, x: Tensor, vel_all: Tensor, edge_index: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        src, dst = edge_index
        N = h.size(0)
        rel   = x[src] - x[dst]
        dist2 = (rel * rel).sum(-1, keepdim=True)
        dist  = dist2.sqrt().clamp(min=1e-8)
        r_hat = rel / dist

        vel_proj_src = (vel_all[src] * r_hat.unsqueeze(1)).sum(-1)
        vel_proj_dst = (vel_all[dst] * r_hat.unsqueeze(1)).sum(-1)

        m_ij = self.phi_e(
            torch.cat([h[src], h[dst], dist2, vel_proj_src, vel_proj_dst], dim=-1)
        )  # (E, hidden)

        if self.update_coords:
            coord_w = self.phi_x(m_ij)
            delta_x = scatter_mean(rel * coord_w, dst, dim=0, dim_size=x.size(0))
            x = x + delta_x

        # === Multi-head softmax attention ===
        # (E, H): per-edge, per-head logit
        e = self.attn(m_ij)
        # Subtract per-(dst,head) max before exp for numerical stability.
        e_max = scatter_max(e, dst, dim=0, dim_size=N)[0]                 # (N, H)
        e_exp = (e - e_max[dst]).exp()                                    # (E, H)
        e_sum = scatter_add(e_exp, dst, dim=0, dim_size=N)                # (N, H)
        alpha = e_exp / (e_sum[dst] + 1e-16)                              # (E, H)

        # Broadcast α across each head's slice of m_ij, then scatter_add.
        # repeat_interleave turns (E, H) → (E, hidden) by replicating each head's
        # weight across its head_dim channels: [α1,α1,...,α1, α2,α2,...α2, ...].
        alpha_wide = alpha.repeat_interleave(self.head_dim, dim=1)        # (E, hidden)
        agg = scatter_add(alpha_wide * m_ij, dst, dim=0, dim_size=N)      # (N, hidden)

        h = h + self.drop(self.phi_h(torch.cat([h, agg], dim=-1)))
        h = self.norm(h)
        return h, x, m_ij


class FixedEGNNAttnModel(FixedEGNNModel):
    """FixedEGNN with multi-head softmax attention in every layer."""

    # Default number of attention heads. Pick heads=1 to recover the
    # single-head softmax variant for ablation against the scalar gate.
    heads = 4

    def __init__(self, features=None, depth=None, hidden_dim=None, dropout=None,
                 update_coords=None, no_slip_mask=None, heads=None, **kwargs):
        super().__init__(
            features=features, depth=depth, hidden_dim=hidden_dim,
            dropout=dropout, update_coords=update_coords, no_slip_mask=no_slip_mask,
            **kwargs,
        )
        if heads is not None:
            self.heads = heads
        # Replace the plain FixedEGNNLayers with multi-head attention layers.
        self.layers = nn.ModuleList([
            FixedEGNNAttnLayer(
                self.hidden_dim,
                dropout=self.dropout,
                update_coords=self.update_coords,
                heads=self.heads,
            )
            for _ in range(self.depth)
        ])
        self._init_weights()
