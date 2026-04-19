"""
FixedEGNNGatedTconv — FixedEGNNGated + 1D temporal conv on velocity magnitudes.

Motivation (2026-04-15): the base model passes 5 velocity magnitudes per point
directly into the node encoder as 5 separate scalars. The model has to rebuild
temporal structure (acceleration, jerk, phase) from these independent numbers
via the downstream layers. A cheap inductive bias: apply a 1D conv across the
5-frame time axis, per-point, so the model gets learned "velocity-derivative"
features for free.

Inspired by the delta / delta-delta features used in classic speech pipelines
(MFCC → Δ → ΔΔ) and by Time-Delay Neural Networks (Waibel et al. 1989): the
conv IS a learned version of the hand-engineered temporal derivative filters.
Modern terminology: a tiny per-point 1D TDNN over the 5-frame velocity history.

Equivariance: the conv acts ONLY on the scalar magnitudes `|v|`, which are by
construction invariant to rotations/translations. So the tconv output is still
an invariant scalar feature. Equivariance of the overall EGNN is preserved.

Cost: one Conv1d with kernel=3, in=1, out=tconv_hidden. For tconv_hidden=3 and
5 input frames → 15 output channels instead of 5. node_encoder's input_dim
grows from 7 → 17. Total extra params: ~10 (conv) + (17-7)·hidden (encoder) = ~325
for hidden=32. Negligible.

Stacks with the gate from FixedEGNNGated — the gate is on edges/messages, the
tconv is on node input. Orthogonal changes.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from models.fixed_egnn_gated.model import FixedEGNNGatedModel


class FixedEGNNGatedTconvModel(FixedEGNNGatedModel):
    """FixedEGNNGated with a learned 1D temporal conv over the 5-frame velocity history."""

    # Hyperparameters for the temporal conv.
    # kernel=3 sees (frame t-1, t, t+1); padding=1 preserves the 5-frame length.
    # tconv_hidden=3 gives 3 learned temporal filters per point — enough to span
    # identity / Δ / ΔΔ if that's what's useful, but data-driven.
    tconv_hidden = 3
    tconv_kernel = 3

    def __init__(self, features=None, depth=None, hidden_dim=None, dropout=None,
                 update_coords=None, no_slip_mask=None, **kwargs):
        super().__init__(
            features=features, depth=depth, hidden_dim=hidden_dim,
            dropout=dropout, update_coords=update_coords, no_slip_mask=no_slip_mask,
            **kwargs,
        )
        # 1D temporal conv over the 5-frame |v| sequence per point.
        # Input shape (B*N, in_channels=1, length=5); output (B*N, tconv_hidden, 5).
        self.tconv = nn.Conv1d(
            in_channels=1,
            out_channels=self.tconv_hidden,
            kernel_size=self.tconv_kernel,
            padding=self.tconv_kernel // 2,   # 'same' padding → preserves length 5
        )

        # Node encoder input grows: tconv_hidden · 5 (temporal features) + 2 (udf).
        # Old was 5 (vel_mag) + 2 = 7.
        new_input_dim = self.tconv_hidden * 5 + 2
        self.node_encoder = nn.Sequential(
            nn.Linear(new_input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
        )
        # Re-apply weight init to the rebuilt encoder + (gated) layers.
        self._init_weights()

    def _egnn_step(
        self,
        pos_flat: Tensor,
        vel_window: Tensor,
        point_features: Tensor | None,
        edge_index: Tensor,
    ) -> Tensor:
        B, _, N, _ = vel_window.shape

        # Per-point velocity magnitudes: (B, N, 5).
        vel_mag = vel_window.norm(dim=-1).transpose(1, 2)

        # Temporal conv: each point is an independent length-5 sequence.
        # (B, N, 5) → (B*N, 1, 5) → conv → (B*N, tconv_hidden, 5) → (B, N, tconv_hidden*5).
        vel_seq = vel_mag.reshape(B * N, 1, 5)
        vel_temporal = self.tconv(vel_seq)                       # (B*N, tconv_hidden, 5)
        vel_temporal = vel_temporal.reshape(B, N, -1)            # (B, N, tconv_hidden*5)

        # Build the scalar node features using the temporal-conv output instead
        # of raw vel_mag. UDF features attach as before.
        if point_features is not None:
            udf_trunc    = point_features[..., 0:1]                        # (B, N, 1)
            udf_grad_mag = point_features[..., 1:4].norm(dim=-1, keepdim=True)
            scalars = torch.cat([vel_temporal, udf_trunc, udf_grad_mag], dim=-1)
        else:
            zeros = torch.zeros(B, N, 2, device=vel_window.device, dtype=vel_window.dtype)
            scalars = torch.cat([vel_temporal, zeros], dim=-1)

        h = self.node_encoder(scalars.reshape(B * N, -1))        # (B*N, hidden)

        # Vector channel (unchanged): per-point velocity history, used inside
        # layers/decoder via projections onto r̂ (equivariant).
        vel_all_flat = vel_window.permute(0, 2, 1, 3).reshape(B * N, 5, 3)

        # Run through the gated EGNN layers and decoder (inherited behavior).
        x = pos_flat.clone() if self.update_coords else pos_flat
        m_ij_last = None
        for layer in self.layers:   # self.layers is FixedEGNNGatedLayer from parent
            h, x, m_ij_last = layer(h, x, vel_all_flat, edge_index)

        delta = self.decoder(h, m_ij_last, x, vel_all_flat, edge_index)
        return delta.reshape(B, N, 3)
