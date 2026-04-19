"""
EGNN (Satorras et al. 2021) implemented from scratch on top of torch_scatter
(part of the torch_geometric ecosystem). Strict E(n)-equivariant.

Mirrors GATModel's structure: Encode-Process-Decode + autoregressive 5-step
rollout. Uses the same FEATURES default and the same input/output shapes.

Per node, per step:
    vel_window_flattened  (15,)  5 timesteps × 3 components
    udf_truncated          (1,)
    udf_gradient           (3,)
    ──────────────────────
    total                 (19,)

Coord updates ARE included (full EGNN). Coords are reset to original `pos`
after each rollout step so the k-NN graph topology stays valid across steps.
"""

import math
import os

import torch
import torch.nn as nn
from torch import Tensor

from models.base import ResidualModel

try:
    from torch_scatter import scatter_add, scatter_mean
    _SCATTER_AVAILABLE = True
except ImportError:
    _SCATTER_AVAILABLE = False


class EGNNLayer(nn.Module):
    """One E(n)-equivariant message-passing layer with coord + node updates."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0, update_coords: bool = True):
        super().__init__()
        self.update_coords = update_coords
        self.phi_e = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.phi_x = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.phi_h = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, h: Tensor, x: Tensor, edge_index: Tensor,vel) -> tuple[Tensor, Tensor]:
        src, dst = edge_index
        rel = x[src] - x[dst]                              # (E, 3)
        dist2 = (rel * rel).sum(-1, keepdim=True)          # (E, 1)
        dist=dist2.sqrt().clamp(min=1e8)
        r_hat=rel /dist                                    # Unit vector (E,3) 

        m_ij = self.phi_e(torch.cat([h[src], h[dst], dist2], dim=-1))  # (E, hidden)

        vel_proj_src=(vel[src])

        # ── Coord update (the equivariant part) ──
        if self.update_coords:
            coord_w = self.phi_x(m_ij)                         # (E, 1)
            delta_x = scatter_mean(rel * coord_w, dst, dim=0, dim_size=x.size(0))
            x = x + delta_x

        # ── Node update ──
        agg = scatter_add(m_ij, dst, dim=0, dim_size=h.size(0))   # (N, hidden)
        h = h + self.drop(self.phi_h(torch.cat([h, agg], dim=-1)))
        h = self.norm(h)
        return h, x


class EGNNModel(ResidualModel):

    FEATURES = ["udf_truncated", "udf_gradient", "knn_graph"]

    hidden_dim    = 32
    depth         = 2
    dropout       = 0.0
    update_coords = False

    def __init__(self, features=None, depth=None, hidden_dim=None, dropout=None, update_coords=None, **kwargs):
        if not _SCATTER_AVAILABLE:
            raise ImportError(
                "torch_scatter is required for EGNNModel. "
                "Install via the Modal image (already pinned in src/modal_image.py)."
            )
        if features   is not None: self.FEATURES   = features
        if depth      is not None: self.depth      = depth
        if hidden_dim is not None: self.hidden_dim = hidden_dim
        if dropout    is not None: self.dropout    = dropout
        if update_coords is not None: self.update_coords = update_coords
        super().__init__()

        node_input_dim = 15 + self.feature_dim

        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList([
            EGNNLayer(self.hidden_dim, dropout=self.dropout, update_coords=self.update_coords) for _ in range(self.depth)
        ])
        self.decoder = nn.Linear(self.hidden_dim, 3)

        self._init_weights()

        weights_path = os.path.join(os.path.dirname(__file__), "state_dict.pt")
        if os.path.exists(weights_path):
            self.load_state_dict(torch.load(weights_path, weights_only=True))

    def _init_weights(self):
        depth_scale = 1.0 / math.sqrt(self.depth) if self.depth > 1 else 1.0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data *= depth_scale
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init decoder → Δv = 0 at init → output = last input frame
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def _build_edge_index(self, knn_graph: Tensor) -> Tensor:
        """Convert (B, N, k) knn_graph (with -1 padding) to global (2, E)."""
        B, N, k = knn_graph.shape
        device = knn_graph.device
        batch_offsets = torch.arange(B, device=device) * N
        rows = torch.arange(N, device=device).view(1, N, 1).expand(B, N, k)
        rows = rows + batch_offsets.view(B, 1, 1)
        cols = knn_graph + batch_offsets.view(B, 1, 1)
        valid = knn_graph >= 0
        return torch.stack([rows[valid], cols[valid]], dim=0)

    def _egnn_step(
        self,
        pos_flat: Tensor,
        vel_window: Tensor,
        point_features: Tensor | None,
        edge_index: Tensor,
    ) -> Tensor:
        B, N = vel_window.shape[0], vel_window.shape[2]
        vel_flat = vel_window.transpose(1, 2).reshape(B, N, 15)
        parts = [vel_flat]
        if point_features is not None:
            parts.append(point_features)
        node_in = torch.cat(parts, dim=-1).reshape(B * N, -1)

        h = self.node_encoder(node_in)
        x = pos_flat.clone()  # coords mutate; original pos stays untouched

        for layer in self.layers:
            h, x = layer(h, x, edge_index)

        return self.decoder(h).reshape(B, N, 3)

    def _predict_delta(self, *args, **kwargs):
        raise NotImplementedError("EGNNModel uses forward() directly.")

    def forward(
        self,
        t: Tensor,
        pos: Tensor,
        idcs_airfoil: list[Tensor],
        velocity_in: Tensor,
        point_features: Tensor | None = None,
        knn_graph: Tensor | None = None,
    ) -> Tensor:
        if self.FEATURES and (point_features is None or knn_graph is None):
            cpf, ckg = self._compute_batch_features(pos, idcs_airfoil)
            if point_features is None: point_features = cpf
            if knn_graph      is None: knn_graph      = ckg

        edge_index = self._build_edge_index(knn_graph)
        B, N = pos.shape[:2]
        pos_flat = pos.reshape(B * N, 3)

        window = velocity_in.clone()
        preds = []
        for _ in range(5):
            delta    = self._egnn_step(pos_flat, window, point_features, edge_index)
            next_vel = window[:, -1] + delta
            preds.append(next_vel.unsqueeze(1))
            window = torch.cat([window[:, 1:], next_vel.unsqueeze(1)], dim=1)
        return torch.cat(preds, dim=1)
