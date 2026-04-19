"""
EGNN wrapper around lucidrains' egnn-pytorch library.

Same I/O contract as EGNNModel (model.py) so they're directly comparable in
the same training loop. Reuses the autoregressive 5-step rollout and the
existing knn_graph from the dataloader.

The library's `EGNN_Sparse` layer takes a flat (N, 3+feat) tensor + edge_index
and returns the same shape, with coords already updated internally.

Install (already pinned in src/modal_image.py):
    pip install egnn-pytorch
"""

import math
import os

import torch
import torch.nn as nn
from torch import Tensor

from models.base import ResidualModel

try:
    from egnn_pytorch import EGNN_Sparse
    _EGNN_LIB_AVAILABLE = True
except ImportError:
    _EGNN_LIB_AVAILABLE = False


class EGNNLibModel(ResidualModel):

    FEATURES = ["udf_truncated", "udf_gradient", "adaptive_knn_graph"]

    hidden_dim    = 32
    depth         = 4
    dropout       = 0.1
    update_coords = True

    def __init__(self, features=None, depth=None, hidden_dim=None, dropout=None, update_coords=None, **kwargs):
        if not _EGNN_LIB_AVAILABLE:
            raise ImportError(
                "egnn-pytorch is required for EGNNLibModel. "
                "It is pinned in src/modal_image.py — rebuild the Modal image."
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
        # EGNN_Sparse expects feats_dim = scalar feature dim (excluding coords)
        self.layers = nn.ModuleList([
            EGNN_Sparse(
                feats_dim=self.hidden_dim,
                m_dim=self.hidden_dim,
                dropout=self.dropout,
                update_coors=self.update_coords,
                update_feats=True,
                norm_feats=True,
            )
            for _ in range(self.depth)
        ])
        self.decoder = nn.Linear(self.hidden_dim, 3)

        self._init_weights()

        weights_path = os.path.join(os.path.dirname(__file__), "state_dict_lib.pt")
        if os.path.exists(weights_path):
            self.load_state_dict(torch.load(weights_path, weights_only=True))

    def _init_weights(self):
        depth_scale = 1.0 / math.sqrt(self.depth) if self.depth > 1 else 1.0
        for m in [self.node_encoder]:
            for sub in m.modules():
                if isinstance(sub, nn.Linear):
                    nn.init.kaiming_normal_(sub.weight, a=0, mode='fan_in', nonlinearity='relu')
                    sub.weight.data *= depth_scale
                    if sub.bias is not None:
                        nn.init.zeros_(sub.bias)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def _build_edge_index(self, knn_graph: Tensor) -> Tensor:
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

        h = self.node_encoder(node_in)               # (B*N, hidden)
        # EGNN_Sparse takes [coords | feats] concatenated
        x = torch.cat([pos_flat, h], dim=-1)          # (B*N, 3 + hidden)

        for layer in self.layers:
            x = layer(x, edge_index)

        h_out = x[:, 3:]                              # strip coords
        return self.decoder(h_out).reshape(B, N, 3)

    def _predict_delta(self, *args, **kwargs):
        raise NotImplementedError("EGNNLibModel uses forward() directly.")

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
