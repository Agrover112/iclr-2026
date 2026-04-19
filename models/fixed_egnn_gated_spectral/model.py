"""
FixedEGNNGatedSpectral — Gated EGNN + EGNO-inspired temporal SpectralConv.

Motivation (2026-04-15):
The current autoregressive rollout's worst observable failure is front-loaded
drift: val jumps +31% between t=0 and t=1 (error_analysis on sample 1021_1-0
logged 2026-04-15). Classic exposure-bias signature — the 5-frame sliding
window's temporal structure is flattened into a 15-scalar input that the model
has to "re-parse" at every rollout step.

EGNO (Xu, Han, Lou, ..., Leskovec, Ermon, Anandkumar; NeurIPS 2024) fixes the
temporal blindness with a SpectralConv on the TIME axis of per-frame hidden
states. FFT in time is trivially cheap for 5 frames (3 complex modes) and
preserves SE(3) equivariance because time is orthogonal to spatial coords.

What this model does:
  1. Encode each of the 5 input frames INDEPENDENTLY via a small per-frame MLP
     (3 scalars → hidden). Shared weights — EGNO-style.
  2. Stack → (B, N, 5, hidden).
  3. `neuralop.layers.SpectralConv` with n_modes=(3,) on the time dim
     (FFT → learnable complex weights per mode → IFFT). Gives every frame
     access to global temporal context in one shot.
  4. Take the post-spectral last-frame representation (+ mean over time as
     auxiliary signal) → feed into existing gated EGNN layers unchanged.
  5. Decode to Δv, rollout as usual.

The rest of the architecture is identical to `FixedEGNNGatedModel`:
  - FixedEGNNGatedLayer with per-edge sigmoid gating (scalable O(E))
  - Zero-init Δv decoder (fix_v2_linear init, depth_scale=1/√depth)
  - No-slip volume mask at airfoil points
  - Autoregressive 5-step rollout (one-shot variant left for a future try)

Scalability / simplicity (GRaM theme):
  - SpectralConv adds ~hidden² · n_modes · 2 params (a few thousand for h=32)
  - O(N · hidden · 5) compute — linear in N, trivial in time dimension
  - No grid construction, no spatial-FFT rasterization (unlike GINO)

Equivariance: preserved. SpectralConv is applied to invariant scalar hidden
features (|v|, |UDF|, UDF gradient magnitude) — not to velocity vectors —
and the FFT is along the time axis, not spatial.
"""

import torch
import torch.nn as nn
from torch import Tensor

try:
    from neuralop.layers.spectral_convolution import SpectralConv
    _NEURALOP_AVAILABLE = True
except ImportError:
    _NEURALOP_AVAILABLE = False

from models.fixed_egnn_gated.model import FixedEGNNGatedModel


class FixedEGNNGatedSpectralModel(FixedEGNNGatedModel):
    """Gated EGNN with EGNO-style temporal SpectralConv on per-frame hidden states."""

    # Per-frame scalar input: vel_mag (1) + udf_truncated (1) + |udf_grad| (1) = 3.
    per_frame_input_dim = 3
    # FFT of 5 real-valued frames has 5//2 + 1 = 3 complex modes.
    n_spectral_modes = 3

    def __init__(
        self,
        features=None,
        depth=None,
        hidden_dim=None,
        dropout=None,
        update_coords=None,
        no_slip_mask=None,
        **kwargs,
    ):
        if not _NEURALOP_AVAILABLE:
            raise ImportError(
                "neuralop is required for FixedEGNNGatedSpectralModel. "
                "Install with `pip install neuraloperator`."
            )
        super().__init__(
            features=features,
            depth=depth,
            hidden_dim=hidden_dim,
            dropout=dropout,
            update_coords=update_coords,
            no_slip_mask=no_slip_mask,
            **kwargs,
        )

        # Per-frame scalar encoder (replaces the multi-frame node_encoder from
        # parent). Same architecture as the parent's node_encoder but 3-dim input.
        self.per_frame_encoder = nn.Sequential(
            nn.Linear(self.per_frame_input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
        )

        # Temporal SpectralConv — from neuralop. Acts on (B*N, hidden, 5).
        # n_modes=(3,) keeps all available Fourier modes for a length-5 sequence.
        self.temporal_spectral = SpectralConv(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            n_modes=(self.n_spectral_modes,),
        )

        # Re-init per-frame encoder with the same kaiming-normal * depth_scale
        # convention the rest of the model uses (see FixedEGNNModel._init_weights).
        self._init_weights()

    def _egnn_step(
        self,
        pos_flat: Tensor,
        vel_window: Tensor,       # (B, 5, N, 3)
        point_features: Tensor | None,
        edge_index: Tensor,
    ) -> Tensor:
        B, T, N, _ = vel_window.shape    # T = 5

        # ── Per-frame scalar features: [vel_mag, udf_trunc, udf_grad_mag] ──
        vel_mag = vel_window.norm(dim=-1)                    # (B, T, N)
        vel_mag = vel_mag.permute(0, 2, 1).contiguous()      # (B, N, T)

        if point_features is not None:
            udf_trunc = point_features[..., 0:1]                                # (B, N, 1)
            udf_grad_mag = point_features[..., 1:4].norm(dim=-1, keepdim=True)  # (B, N, 1)
        else:
            udf_trunc = torch.zeros(B, N, 1, device=vel_window.device, dtype=vel_window.dtype)
            udf_grad_mag = torch.zeros_like(udf_trunc)

        # Broadcast UDF across T frames and stack with vel_mag.
        # Shape per frame: (B, N, 3) = [vel_mag_t, udf_trunc, udf_grad_mag]
        vel_mag_per_frame = vel_mag.unsqueeze(-1)                          # (B, N, T, 1)
        udf_t_per_frame = udf_trunc.unsqueeze(2).expand(-1, -1, T, -1)     # (B, N, T, 1)
        udf_g_per_frame = udf_grad_mag.unsqueeze(2).expand(-1, -1, T, -1)  # (B, N, T, 1)
        per_frame = torch.cat(
            [vel_mag_per_frame, udf_t_per_frame, udf_g_per_frame], dim=-1
        )                                                                   # (B, N, T, 3)

        # ── Encode each (point, frame) pair independently (shared weights) ──
        h_pf = self.per_frame_encoder(
            per_frame.reshape(B * N * T, self.per_frame_input_dim)
        )                                                                    # (B*N*T, hidden)
        H = h_pf.reshape(B * N, T, self.hidden_dim)                          # (B*N, T, hidden)

        # ── EGNO-style temporal SpectralConv ──
        # neuralop.SpectralConv expects (batch, in_channels, sequence). Transpose.
        H_spec = self.temporal_spectral(H.transpose(1, 2))                   # (B*N, hidden, T)
        H_spec = H_spec.transpose(1, 2)                                      # (B*N, T, hidden)

        # Aggregate temporal dim. Use the LAST frame's spectral-enriched rep
        # (most physically meaningful: "state at t=current, informed by the
        # full 5-frame history via FFT") plus the temporal MEAN as additional
        # global context, summed together to avoid doubling hidden_dim.
        h = H_spec[:, -1, :] + H_spec.mean(dim=1)                            # (B*N, hidden)

        # ── Vector channel: per-point 5-frame velocity history, as before ──
        vel_all_flat = vel_window.permute(0, 2, 1, 3).reshape(B * N, T, 3)

        # ── Gated EGNN layers (inherited) and decoder (inherited) ──
        x = pos_flat.clone() if self.update_coords else pos_flat
        m_ij_last = None
        for layer in self.layers:
            h, x, m_ij_last = layer(h, x, vel_all_flat, edge_index)

        delta = self.decoder(h, m_ij_last, x, vel_all_flat, edge_index)
        return delta.reshape(B, N, 3)
