"""
FixedEGNNRecurrent — FixedEGNN with a GRUCell carrying hidden state across the
5 autoregressive rollout steps.

Motivation (from Remi's suggestion on 2026-04-14): the base `fixed_egnn` re-encodes
the velocity window from scratch at every rollout step and carries ZERO memory
between steps except via the sliding window itself. This likely contributes to
drift at later timesteps, since each step has to rebuild its understanding of
the flow state from only 5 velocity frames + UDF.

Light latent recurrence: after the GNN layers produce a per-node hidden state
`h_t`, blend it with the previous step's state `h_{t-1}` via a `nn.GRUCell` before
the decoder runs. The GRU's reset/update gates let the model choose how much
prior state to keep vs. let the new observation override. At step 0 we use a
zero init for `h_{-1}`, matching the usual RNN convention.

Cost: 3 · hidden² extra params (for h=32: 3072 params, tiny). Compute is O(N·hidden)
per rollout step — no graph operation, trivial overhead. Scalability preserved.

Theme fit (GRaM "simplicity + scalability"):
  - No O(N²) attention, no neural ODE, no multi-scale refactor.
  - Single GRUCell, ~50 LOC diff vs `fixed_egnn`.
  - Addresses a concrete failure mode (rollout drift) with a well-understood mechanism.
"""

import torch
import torch.nn as nn
from torch import Tensor

from models.fixed_egnn.model import FixedEGNNModel


class FixedEGNNRecurrentModel(FixedEGNNModel):
    """FixedEGNN with a GRU carrying per-node hidden state across the 5 rollout steps."""

    def __init__(self, features=None, depth=None, hidden_dim=None, dropout=None,
                 update_coords=None, no_slip_mask=None, **kwargs):
        super().__init__(
            features=features, depth=depth, hidden_dim=hidden_dim,
            dropout=dropout, update_coords=update_coords, no_slip_mask=no_slip_mask,
            **kwargs,
        )
        # GRU cell mixes (h_new_from_encoder_and_layers, h_from_prev_rollout_step)
        # → h_combined used by decoder. Initial hidden state for step 0 is zeros.
        self.gru_cell = nn.GRUCell(self.hidden_dim, self.hidden_dim)

    def _egnn_step_with_state(
        self,
        pos_flat: Tensor,
        vel_window: Tensor,
        point_features: Tensor | None,
        edge_index: Tensor,
        h_prev: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Same as FixedEGNN._egnn_step but returns (delta, h_combined).

        h_combined is the GRU-blended hidden state that goes to the decoder AND
        is carried to the next rollout step as h_prev.
        """
        B, _, N, _ = vel_window.shape

        # Node scalar features (5 vel mags + udf_trunc + |udf_grad|) — same as parent.
        vel_mag = vel_window.norm(dim=-1).transpose(1, 2)              # (B, N, 5)
        if point_features is not None:
            udf_trunc    = point_features[..., 0:1]                    # (B, N, 1)
            udf_grad_mag = point_features[..., 1:4].norm(dim=-1, keepdim=True)
            scalars = torch.cat([vel_mag, udf_trunc, udf_grad_mag], dim=-1)
        else:
            zeros = torch.zeros(B, N, 2, device=vel_window.device, dtype=vel_window.dtype)
            scalars = torch.cat([vel_mag, zeros], dim=-1)

        h = self.node_encoder(scalars.reshape(B * N, -1))              # (B*N, hidden)

        vel_all_flat = vel_window.permute(0, 2, 1, 3).reshape(B * N, 5, 3)
        x = pos_flat.clone() if self.update_coords else pos_flat

        m_ij_last = None
        for layer in self.layers:
            h, x, m_ij_last = layer(h, x, vel_all_flat, edge_index)

        # ── Recurrent latent state: GRU between rollout steps ─────────────
        # At step 0, h_prev is None → initialize to zeros (standard RNN convention).
        if h_prev is None:
            h_prev = torch.zeros_like(h)
        h_combined = self.gru_cell(h, h_prev)                          # (B*N, hidden)

        delta = self.decoder(h_combined, m_ij_last, x, vel_all_flat, edge_index)
        return delta.reshape(B, N, 3), h_combined

    def forward(
        self,
        t: Tensor,
        pos: Tensor,
        idcs_airfoil: list[Tensor],
        velocity_in: Tensor,
        point_features: Tensor | None = None,
        knn_graph: Tensor | None = None,
    ) -> Tensor:
        # Feature fallback (same as parent).
        if self.FEATURES and (point_features is None or knn_graph is None):
            cpf, ckg = self._compute_batch_features(pos, idcs_airfoil)
            if point_features is None: point_features = cpf
            if knn_graph      is None: knn_graph      = ckg

        edge_index = self._build_edge_index(knn_graph)
        B, N = pos.shape[:2]
        pos_flat = pos.reshape(B * N, 3)

        # No-slip mask (same as parent).
        if self.no_slip_mask:
            vol_mask = torch.ones(B, N, 1, device=pos.device, dtype=velocity_in.dtype)
            for b, idc in enumerate(idcs_airfoil):
                vol_mask[b, idc, :] = 0.0
        else:
            vol_mask = None

        window = velocity_in.clone()
        preds = []
        h_prev: Tensor | None = None  # GRU initial state = zeros on first step

        for _ in range(5):
            delta, h_prev = self._egnn_step_with_state(
                pos_flat, window, point_features, edge_index, h_prev,
            )
            next_vel = window[:, -1] + delta
            if vol_mask is not None:
                next_vel = next_vel * vol_mask
            preds.append(next_vel.unsqueeze(1))
            window = torch.cat([window[:, 1:], next_vel.unsqueeze(1)], dim=1)

        return torch.cat(preds, dim=1)
