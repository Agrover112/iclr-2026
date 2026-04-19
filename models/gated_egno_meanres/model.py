"""
GatedEGNOMeanRes — GatedEGNO with mean-reference residual instead of last-frame.

Standard residual:   v_out = v_in[-1] + delta
Mean residual:       v_out = mean(v_in) + delta

Motivation (Reynolds decomposition): the last input frame v_in[-1] includes
the instantaneous turbulent fluctuation at one moment. Using the 5-frame
temporal mean as the residual reference removes most of the oscillation from
the reference itself, giving a cleaner delta target with lower variance.

In the wake, v_in[-1] might catch a vortex at some random phase — the delta
has to "undo" that phase AND predict the future. With v̄, the reference is
phase-averaged, so delta only needs to predict the deviation from the mean
flow, which is a smoother, more learnable target.

Architecture is identical to GatedEGNOModel — only line 379 changes:
    last_frame = velocity_in[:, -1:]        → ORIGINAL
    mean_frame = velocity_in.mean(dim=1, keepdim=True)  → THIS MODEL
"""

import torch
from torch import Tensor

from models.gated_egno.model import GatedEGNOModel, sinusoidal_time_embedding


class GatedEGNOMeanResModel(GatedEGNOModel):
    """GatedEGNO with temporal-mean residual reference."""

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

        B, T, N, _ = velocity_in.shape
        BN = B * N
        edge_index = self._build_edge_index(knn_graph)
        pos_flat = pos.reshape(BN, 3)

        # ── Per-frame scalar features ──
        vel_mag = velocity_in.norm(dim=-1).transpose(1, 2)
        if point_features is not None:
            udf_t = point_features[..., 0:1]
            udf_g = point_features[..., 1:4].norm(dim=-1, keepdim=True)
        else:
            udf_t = torch.zeros(B, N, 1, device=pos.device, dtype=velocity_in.dtype)
            udf_g = torch.zeros_like(udf_t)

        vm = vel_mag.unsqueeze(-1)
        ut = udf_t.unsqueeze(2).expand(-1, -1, T, -1)
        ug = udf_g.unsqueeze(2).expand(-1, -1, T, -1)
        per_frame = torch.cat([vm, ut, ug], dim=-1)
        per_frame = per_frame.permute(2, 0, 1, 3).reshape(T, BN, 3)

        t_emb = sinusoidal_time_embedding(
            torch.arange(T, device=pos.device), self.t_emb_dim,
        )
        t_emb = t_emb.unsqueeze(1).expand(T, BN, self.t_emb_dim)

        h_in = torch.cat([per_frame, t_emb], dim=-1)
        h = self.per_frame_encoder(h_in.reshape(T * BN, -1)).reshape(T, BN, self.hidden_dim)

        vel_all = velocity_in.permute(0, 2, 1, 3).reshape(BN, T, 3)

        for block in self.blocks:
            h, vel_all = block(h, pos_flat, vel_all, edge_index)

        E = edge_index.shape[1]
        h_batched = h.reshape(T * BN, self.hidden_dim)
        x_batched = pos_flat.repeat(T, 1)
        vel_all_batched = vel_all.repeat(T, 1, 1)
        offsets = (torch.arange(T, device=pos.device) * BN).repeat_interleave(E)
        edges_batched = edge_index.repeat(1, T) + offsets.unsqueeze(0)

        last_gnn = self.blocks[-1].gnn
        _, _, m_ij_last = last_gnn(h_batched, x_batched, vel_all_batched, edges_batched)
        delta_flat = self.decoder(
            h_batched, m_ij_last, x_batched, vel_all_batched, edges_batched,
        )
        delta = delta_flat.reshape(T, B, N, 3).permute(1, 0, 2, 3).contiguous()

        # ── Mean-reference residual (the ONLY change from GatedEGNOModel) ──
        mean_frame = velocity_in.mean(dim=1, keepdim=True)        # (B, 1, N, 3)
        v_out = mean_frame + delta                                 # (B, T, N, 3)

        # ── No-slip mask ──
        if self.no_slip_mask:
            vol_mask = torch.ones(B, N, 1, device=pos.device, dtype=velocity_in.dtype)
            for b, idc in enumerate(idcs_airfoil):
                vol_mask[b, idc, :] = 0.0
            v_out = v_out * vol_mask.unsqueeze(1)

        return v_out
