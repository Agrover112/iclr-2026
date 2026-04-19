"""
ResidualMLP — MLP that predicts velocity residuals with geometric features.

Input per point:
    pos           (3,)   x, y, z coordinates
    velocity_in  (15,)   5 timesteps × 3 components, flattened
    udf_truncated (1,)   distance to nearest airfoil point, capped at 0.5
    udf_gradient  (3,)   unit vector toward nearest surface point
    ─────────────────
    total        (22,)

Output per point:
    delta        (15,)   5 timesteps × 3 components of velocity change
    (base class adds velocity_in[:, -1:] → absolute velocity)
"""

import math
import os

import torch
import torch.nn as nn

from models.base import ResidualModel
from src.features import total_feature_dim


class ResidualMLP(ResidualModel):

    FEATURES = ["udf_truncated", "udf_gradient", "local_density"]   # default features

    # output = 5 timesteps × 3 = 15
    hidden_dim = 256
    output_dim = 15
    dropout_probability = 0.1

    def __init__(self, features=None):
        # Allow overriding features before super().__init__() computes feature_dim
        if features is not None:
            self.FEATURES = features
        super().__init__()

        # input = pos(3) + vel_in(15) + feature_dim (dynamic)
        input_dim = 3 + 15 + self.feature_dim
        num_channels = (input_dim, self.hidden_dim, self.hidden_dim, self.output_dim)

        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activations = nn.ModuleList()

        for ch_in, ch_out in zip(num_channels[:-2], num_channels[1:-1]):
            self.linears.append(nn.Linear(ch_in, ch_out))
            self.norms.append(nn.LayerNorm(ch_out))
            self.activations.append(nn.ReLU())

        self.linears.append(nn.Linear(*num_channels[-2:]))
        self.norms.append(nn.Identity())
        self.activations.append(nn.Identity())

        self.dropout = nn.Dropout(self.dropout_probability)

        # Initialize weights to zero delta (= naive baseline at start of training)
        self._init_weights()

        # Load saved weights if available and compatible
        weights_path = os.path.join(
            os.path.dirname(__file__), "state_dict.pt"
        )
        if os.path.exists(weights_path):
            state = torch.load(weights_path, weights_only=True)
            saved_in = state['linears.0.weight'].shape[1]
            if saved_in == self.linears[0].in_features:
                self.load_state_dict(state)
            # Otherwise keep freshly initialized weights

    def _init_weights(self):
        """Kaiming init scaled by depth to prevent gradient vanishing in deep MLPs."""
        depth = len(self.linears)
        depth_scale = 1.0 / math.sqrt(depth) if depth > 1 else 1.0

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data *= depth_scale
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Zero out final layer → delta = 0 at init → output = last input frame
        nn.init.zeros_(self.linears[-1].weight)
        nn.init.zeros_(self.linears[-1].bias)

    def _predict_delta(self, t, pos, idcs_airfoil, velocity_in, point_features, knn_graph=None):
        """Run MLP, predict velocity delta.

        Args:
            pos:            (batch, N, 3)
            velocity_in:    (batch, 5, N, 3)
            point_features: (batch, N, F) — dynamic based on self.FEATURES

        Returns:
            delta: (batch, 5, N, 3)
        """
        batch_size, num_t_in, num_pos, _ = velocity_in.shape

        # Flatten velocity over time: (batch, N, 5*3=15)
        vel_flat = velocity_in.transpose(1, 2).reshape(batch_size, num_pos, num_t_in * 3)

        # Concatenate all per-point inputs: (batch, N, 3+15+F)
        parts = [pos, vel_flat]
        if point_features is not None and point_features.shape[-1] > 0:
            parts.append(point_features)
        x = torch.cat(parts, dim=2)

        # MLP forward
        for linear, norm, activation in zip(self.linears, self.norms, self.activations):
            x = activation(norm(linear(self.dropout(x))))

        # Reshape to (batch, 5, N, 3)
        return x.view(batch_size, num_pos, num_t_in, 3).transpose(1, 2)
