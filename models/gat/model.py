"""
GATv2 with autoregressive rollout — GNS-inspired Encode-Process-Decode architecture.

Predicts 1 velocity timestep at a time (velocity delta Δv), sliding a 5-step
context window forward. Unrolled 5× to produce (batch, 5, N, 3) output.

Input per node (per step):
    vel_window_flattened  (15,)  5 timesteps × 3 components
    udf_truncated          (1,)  distance to airfoil, capped at 0.5
    udf_gradient           (3,)  unit vector toward nearest surface point
    ──────────────────────────
    total                 (19,)

Edge features:
    [Δx, Δy, Δz, dist]   (4,)  relative displacement + distance

Architecture: Encoder MLP → N × GATv2Conv (unshared) → Decoder MLP → Δv
"""

import math
import os

import torch
import torch.nn as nn
from torch import Tensor

from models.base import ResidualModel

try:
    from torch_geometric.nn import GATv2Conv
    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False


class GATModel(ResidualModel):

    FEATURES = ["udf_truncated", "udf_gradient", "adaptive_knn_graph"]

    hidden_dim = 32
    edge_dim    = 8
    depth       = 8
    heads       = 4
    dropout     = 0.1

    def __init__(self, features=None, depth=None, hidden_dim=None, heads=None, dropout=None):
        if not _PYG_AVAILABLE:
            raise ImportError(
                "torch_geometric is required for GATModel. "
                "Install via: pip install torch_geometric"
            )
        if features is not None:
            self.FEATURES = features
        if depth is not None:
            self.depth = depth
        if hidden_dim is not None:
            self.hidden_dim = hidden_dim
        if heads is not None:
            self.heads = heads
        if dropout is not None:
            self.dropout = dropout
        super().__init__()

        # node_input_dim = vel_flat(15) + float_features (udf_truncated=1, udf_gradient=3)
        node_input_dim = 15 + self.feature_dim
        head_dim = self.hidden_dim // self.heads  # 128 // 4 = 32

        # ── Encoder ──────────────────────────────────────────────────
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, self.edge_dim),   # [Δx, Δy, Δz, dist]
            nn.LayerNorm(self.edge_dim),
            nn.ReLU(),
        )

        # ── Processor: unshared GATv2Conv layers ──────────────────────
        self.convs = nn.ModuleList([
            GATv2Conv(
                self.hidden_dim, head_dim,
                heads=self.heads,
                edge_dim=self.edge_dim,
                dropout=self.dropout,
                concat=True,         # output = heads * head_dim = hidden_dim
                add_self_loops=False,  # edge_attr provided explicitly
            )
            for _ in range(self.depth)
        ])
        self.conv_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.depth)
        ])

        # ── Decoder ───────────────────────────────────────────────────
        self.decoder = nn.Linear(self.hidden_dim, 3)

        self._init_weights()

        # Load saved weights if available
        weights_path = os.path.join(os.path.dirname(__file__), "state_dict.pt")
        if os.path.exists(weights_path):
            self.load_state_dict(torch.load(weights_path, weights_only=True))

    def _init_weights(self):
        """Kaiming init scaled by depth; zero-init decoder for naive baseline (Δv = 0)."""
        depth_scale = 1.0 / math.sqrt(self.depth) if self.depth > 1 else 1.0

        # Initialize encoders and processor layers
        for module in [self.node_encoder, self.edge_encoder, self.conv_norms]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                    m.weight.data *= depth_scale
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # Initialize GATv2Conv layers (they have their own init, but scale them too)
        for conv in self.convs:
            if hasattr(conv, 'lin_l'):
                nn.init.kaiming_normal_(conv.lin_l.weight, a=0, mode='fan_in', nonlinearity='relu')
                conv.lin_l.weight.data *= depth_scale

        # Zero-init decoder → delta = 0 at init → output = last input frame
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def _build_graph(self, pos: Tensor, knn_graph: Tensor) -> tuple[Tensor, Tensor]:
        """Convert batched (B, N, k) knn_graph to PyG edge_index + encoded edge_attr.

        Handles -1 padding in adaptive_knn_graph.

        Returns:
            edge_index: (2, E)  — global node indices across the batch
            edge_attr:  (E, edge_dim)
        """
        B, N, k = knn_graph.shape
        device = pos.device

        batch_offsets = torch.arange(B, device=device) * N  # (B,)

        rows = torch.arange(N, device=device).view(1, N, 1).expand(B, N, k)
        rows = rows + batch_offsets.view(B, 1, 1)             # (B, N, k) global src
        cols = knn_graph + batch_offsets.view(B, 1, 1)        # (B, N, k) global dst

        valid = knn_graph >= 0                                 # mask -1 padding
        src = rows[valid]                                      # (E,)
        dst = cols[valid]                                      # (E,)
        edge_index = torch.stack([src, dst], dim=0)            # (2, E)

        pos_flat = pos.reshape(B * N, 3)
        rel_pos  = pos_flat[dst] - pos_flat[src]               # (E, 3) Δx,Δy,Δz
        dist     = rel_pos.norm(dim=1, keepdim=True)           # (E, 1)
        edge_attr = self.edge_encoder(
            torch.cat([rel_pos, dist], dim=1)                  # (E, 4)
        )                                                       # (E, edge_dim)

        return edge_index, edge_attr

    def _gat_step(
        self,
        pos: Tensor,
        vel_window: Tensor,
        point_features: Tensor | None,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        """Single-step prediction: context window → velocity delta.

        Args:
            pos:            (B, N, 3)
            vel_window:     (B, 5, N, 3) — current context window
            point_features: (B, N, F) float features, or None
            edge_index:     (2, E)
            edge_attr:      (E, edge_dim)

        Returns:
            delta: (B, N, 3)
        """
        B, N = pos.shape[:2]

        vel_flat = vel_window.transpose(1, 2).reshape(B, N, 15)  # (B, N, 15)
        parts = [vel_flat]
        if point_features is not None:
            parts.append(point_features)
        node_in = torch.cat(parts, dim=-1).reshape(B * N, -1)    # (B*N, 19)

        x = self.node_encoder(node_in)                            # (B*N, hidden_dim)

        # Processor: residual skip every 2 layers
        for i in range(0, self.depth, 2):
            h = torch.relu(self.conv_norms[i](self.convs[i](x, edge_index, edge_attr)))
            if i + 1 < self.depth:
                h = torch.relu(self.conv_norms[i + 1](self.convs[i + 1](h, edge_index, edge_attr)))
            x = x + h

        return self.decoder(x).reshape(B, N, 3)                   # (B, N, 3)

    def _predict_delta(self, *args, **kwargs):
        raise NotImplementedError("GATModel uses forward() directly.")

    def forward(
        self,
        t: Tensor,
        pos: Tensor,
        idcs_airfoil: list[Tensor],
        velocity_in: Tensor,
        point_features: Tensor | None = None,
        knn_graph: Tensor | None = None,
    ) -> Tensor:
        """Autoregressive rollout: 5 steps, sliding the 5-step context window.

        Each step predicts Δv; window advances with model's own prediction.

        Returns absolute velocity: (B, 5, N, 3)
        """
        if self.FEATURES and (point_features is None or knn_graph is None):
            computed_pf, computed_knn = self._compute_batch_features(pos, idcs_airfoil)
            if point_features is None:
                point_features = computed_pf
            if knn_graph is None:
                knn_graph = computed_knn

        # Build graph once — topology (pos) is fixed across all 5 steps
        edge_index, edge_attr = self._build_graph(pos, knn_graph)

        window = velocity_in.clone()  # (B, 5, N, 3)
        preds = []

        for _ in range(5):
            delta    = self._gat_step(pos, window, point_features, edge_index, edge_attr)
            next_vel = window[:, -1] + delta             # v_{t+1} = v_t + Δv
            preds.append(next_vel.unsqueeze(1))
            window = torch.cat([window[:, 1:], next_vel.unsqueeze(1)], dim=1)

        return torch.cat(preds, dim=1)                   # (B, 5, N, 3)
