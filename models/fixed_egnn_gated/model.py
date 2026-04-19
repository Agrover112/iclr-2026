"""
FixedEGNNGated — FixedEGNN with multi-head learnable gates on edge messages.

Motivation: the base FixedEGNN aggregates messages with plain scatter_add —
every edge contributes equally to each receiver's aggregation. With fixed-k
neighbors on an irregular mesh, this treats a boundary-layer neighbor (high-
shear, highly informative) the same as a far-field neighbor (low-gradient,
nearly redundant).

Multi-head sigmoid gating: split m_ij into H slices of size hidden/H. Each
head produces its own scalar gate g_ij^(h) ∈ (0, 1) and scales its own slice:
    agg_i = concat_h [ Σ_j g_ij^(h) · m_ij^(h) ]
Independent heads can specialize on different features in the message
(wall-normal, streamwise, proximity, shear) without cross-talk. Asymptotic
cost stays O(E·H) with no normalization across neighbors — unlike GAT-style
softmax (see FixedEGNNAttn), gates are purely local per edge.

Parameter cost: one Linear(hidden, H) per layer. For d=6 h=32 H=4 that's
132·6 = 792 extra params. Negligible.

Zero-init: all gate logits = 0 → sigmoid = 0.5 uniformly per head → plain
aggregation with a constant scale absorbed by phi_h. Gradient flows from
step 0 via bias and via h through the phi_h path.

heads=1 recovers the original scalar-gate behavior.
"""

import torch
import torch.nn as nn
from torch import Tensor

try:
    from torch_scatter import scatter_add, scatter_mean
except ImportError:  # pragma: no cover — same fallback pattern as fixed_egnn
    scatter_add = scatter_mean = None

from models.fixed_egnn.model import FixedEGNNModel, FixedEGNNLayer


class FixedEGNNGatedLayer(FixedEGNNLayer):
    """EGNN layer with multi-head sigmoid gating on aggregated messages."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0,
                 update_coords: bool = False, heads: int = 4):
        super().__init__(hidden_dim, dropout=dropout, update_coords=update_coords)
        if hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by heads={heads}"
            )
        self.heads = heads
        self.head_dim = hidden_dim // heads
        # H scalar gates per edge; each head scales its own slice of m_ij.
        # Zero-init weight so sigmoid = 0.5 uniformly per head at init (plain
        # aggregation with a constant scale, absorbed by phi_h).
        self.gate = nn.Linear(hidden_dim, heads)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(
        self, h: Tensor, x: Tensor, vel_all: Tensor, edge_index: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        src, dst = edge_index
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

        # === Multi-head sigmoid gating ===
        # g_ij^(h) ∈ (0, 1): per-edge, per-head gate. Broadcast each head's
        # scalar across its head_dim-slice of m_ij before scatter_add.
        g = torch.sigmoid(self.gate(m_ij))                          # (E, H)
        g_wide = g.repeat_interleave(self.head_dim, dim=1)          # (E, hidden)
        agg = scatter_add(g_wide * m_ij, dst, dim=0, dim_size=h.size(0))

        h = h + self.drop(self.phi_h(torch.cat([h, agg], dim=-1)))
        h = self.norm(h)
        return h, x, m_ij


class FixedEGNNGatedModel(FixedEGNNModel):
    """FixedEGNN with multi-head gated edge aggregation in every layer."""

    # Default number of gating heads. heads=1 recovers the original scalar
    # gate; heads=4 matches the attn variant's default for a clean ladder.
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
        # Replace the plain FixedEGNNLayers the parent constructed with multi-head
        # gated ones. All other architecture (encoder, decoder, init, no-slip) identical.
        self.layers = nn.ModuleList([
            FixedEGNNGatedLayer(
                self.hidden_dim,
                dropout=self.dropout,
                update_coords=self.update_coords,
                heads=self.heads,
            )
            for _ in range(self.depth)
        ])
        # Re-apply the parent's init to the new layer params so scaling stays
        # consistent (kaiming_normal * 1/sqrt(depth)).
        self._init_weights()
