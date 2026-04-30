"""
GatedEGNO — faithful EGNO-style (Xu, Han, Lou, Kossaifi, …, Leskovec, Ermon,
Anandkumar; ICML 2024) with GatedEGNN as the per-time-slice backbone instead
of plain EGNN.

Reference repo: https://github.com/MinkaiXu/egno — we follow the structure in
`model/egno.py` and `model/layer_no.py`:

    for each of n_layers blocks:
        h = h + LeakyReLU(SpectralConv1d_time(h))     # TimeConv
        v = v + SpectralConv1d_x_time(v)              # TimeConv_x  (equivariant)
        h, x, v = EGNN_layer(h, x, v, edges)          # per-time-slice message passing

We replace the plain EGNN layer with our validated `FixedEGNNGatedLayer`
(per-edge sigmoid gating, LayerNorm, residual).

Key differences from the official EGNO:
  * Time axis: we set T = number of output timesteps (5). Input frames are
    processed via the per-frame scalar encoder into h_{t=0..4} BEFORE the
    EGNO blocks. So "time" in the blocks is really "input frame index" and
    we decode each time slice to the corresponding output frame's Δv.
  * Gated message passing: `FixedEGNNGatedLayer` with `g_ij = sigmoid(Linear(m_ij))`
    scaling per-edge messages before scatter_add.
  * One-shot output: all 5 output frames produced in one forward pass, no
    autoregressive rollout. This kills the front-loaded rollout drift
    (31% jump at t=0→t=1 observed in LOG.md 2026-04-15).

Equivariance: preserved. TimeConv on scalar h is invariant. TimeConv_x keeps
the spatial D=3 dimension as a preserved axis (FFT/IFFT mixes only along
time, not across spatial components). FixedEGNNGatedLayer is equivariant.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from torch_scatter import scatter_add, scatter_mean
    _SCATTER_AVAILABLE = True
except ImportError:
    _SCATTER_AVAILABLE = False

try:
    from neuralop.layers.spectral_convolution import SpectralConv
    _NEURALOP_AVAILABLE = True
except ImportError:
    _NEURALOP_AVAILABLE = False

from models.base import ResidualModel, GRAPH_FEATURES, FEATURE_REGISTRY
from models.fixed_egnn.model import FixedEGNNLayer, EquivariantDecoder
from models.fixed_egnn_gated.model import FixedEGNNGatedLayer


def sinusoidal_time_embedding(t: Tensor, dim: int) -> Tensor:
    """Sinusoidal positional embedding (diffusion-style) for timestep indices.

    Args:
        t:  (T,) integer tensor of timestep indices 0..T-1
        dim: embedding dim

    Returns:
        (T, dim) float tensor
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)  # (T, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (T, 2*half)
    if dim % 2:   # pad to exact dim if odd
        emb = F.pad(emb, (0, 1))
    return emb


class TimeConv(nn.Module):
    """Temporal SpectralConv on hidden scalar features. EGNO Eq. below their §3.

    Input/output: (T, BN, C) — T timesteps on the "length" axis, BN spatial
    points flattened into batch, C channels (hidden dim).

    Implements `h + LeakyReLU(SpectralConv1d_time(h))` per EGNO's TimeConv.
    """

    def __init__(self, channels: int, n_modes: int = 3):
        super().__init__()
        # neuralop.SpectralConv with n_modes=(n_modes,) for 1D spectral along time.
        self.spec = SpectralConv(channels, channels, n_modes=(n_modes,))
        self.act = nn.LeakyReLU(0.2)

    def forward(self, h: Tensor) -> Tensor:   # (T, BN, C) → (T, BN, C)
        # SpectralConv expects (batch, channels, length). We have (T, BN, C);
        # reshape to (BN, C, T).
        T, BN, C = h.shape
        x = h.permute(1, 2, 0).contiguous()        # (BN, C, T)
        x = self.spec(x)                            # (BN, C, T)
        x = x.permute(2, 0, 1).contiguous()        # (T, BN, C)
        return h + self.act(x)


class TimeConvX(nn.Module):
    """Equivariant temporal SpectralConv on coords/velocity.

    Input/output: (T, BN, D, C) — D=3 spatial components kept as their own
    axis (never mixed across D by the learned weights) to preserve rotation
    equivariance. Following EGNO's SpectralConv1d_x + TimeConv_x pattern.

    Residual: returns v + spec(v) (no activation — keeps linearity on the
    equivariant channel).

    Channel count C is INDEPENDENT of hidden_dim — the velocity channel is
    just its raw 3D components (C=1). The SpectralConv learns per-mode scalar
    weights that mix across time but not across D. The D=3 spatial axis is
    absorbed into batch (one spectral-conv-worth of parameters shared across D).
    """

    def __init__(self, channels: int, n_modes: int = 3):
        super().__init__()
        # channels here is the per-velocity-component channel count (1 for raw
        # velocity). NOT the main hidden_dim — that's for the scalar h channel.
        self.channels = channels
        self.spec = SpectralConv(channels, channels, n_modes=(n_modes,))

    def forward(self, v: Tensor) -> Tensor:   # (T, BN, D, C) → (T, BN, D, C)
        T, BN, D, C = v.shape
        assert C == self.channels, (
            f"TimeConvX channel mismatch: got {C}, expected {self.channels}"
        )
        # Absorb D into the batch dim so per-D we share spectral weights.
        x = v.permute(1, 2, 3, 0).reshape(BN * D, C, T)    # (BN·D, C, T)
        x = self.spec(x)                                    # (BN·D, C, T) — same shape
        x = x.reshape(BN, D, C, T).permute(3, 0, 1, 2).contiguous()  # (T, BN, D, C)
        return v + x


class GatedEGNOBlock(nn.Module):
    """One EGNO block: TimeConv_h + TimeConv_v + FixedEGNNGatedLayer per time slice.

    Operates on a batched (T, BN) representation where the T time slices share
    the same spatial graph. Message passing is per-slice (a for-loop over T);
    cross-time information flows only through the two TimeConv modules.
    """

    def __init__(self, hidden_dim: int, n_modes: int = 3,
                 dropout: float = 0.0, update_coords: bool = False,
                 heads: int = 1, use_gate: bool = True):
        super().__init__()
        self.time_conv_h = TimeConv(hidden_dim, n_modes=n_modes)
        self.time_conv_v = TimeConvX(channels=1, n_modes=n_modes)
        if use_gate:
            # Gated variant (default) — EGNN with sigmoid edge-inference gate.
            # heads=1 by default — original single-scalar gate per edge. The
            # multi-head extension (heads>1) is from FixedEGNNGatedLayer's
            # newer behaviour; we default to 1 to stay compatible with
            # checkpoints trained before multi-head was introduced.
            self.gnn = FixedEGNNGatedLayer(
                hidden_dim, dropout=dropout,
                update_coords=update_coords, heads=heads,
            )
        else:
            # Ungated ablation: plain FixedEGNNLayer, e_ij ≡ 1. Used to
            # measure whether the gate earns its keep inside the full EGNO
            # stack (spectral time mixing + mean-residual decoder).
            from models.fixed_egnn.model import FixedEGNNLayer
            self.gnn = FixedEGNNLayer(
                hidden_dim, dropout=dropout, update_coords=update_coords,
            )

    def forward(
        self,
        h: Tensor,              # (T, BN, C) — hidden scalars over time
        x: Tensor,              # (BN, 3) — fixed coords (Eulerian mesh)
        vel_all: Tensor,        # (BN, T, 3) — per-point velocity history
        edge_index: Tensor,     # (2, E) — single-slice graph (shared across T)
    ) -> tuple[Tensor, Tensor]:
        T, BN, C = h.shape

        # ── Temporal mixing on scalar h ──
        h = self.time_conv_h(h)        # (T, BN, C)

        # ── Temporal mixing on velocity (equivariant) ──
        # Prepare v as (T, BN, 3, 1) — one channel (the velocity itself).
        # For now C_vel = 1: just the raw 3D velocity per time. If we wanted
        # to learn richer temporal velocity representations, expand C_vel > 1.
        vel_td = vel_all.transpose(0, 1).unsqueeze(-1)   # (T, BN, 3, 1)
        vel_td = self.time_conv_v(vel_td)               # (T, BN, 3, 1)
        vel_all_new = vel_td.squeeze(-1).transpose(0, 1).contiguous()  # (BN, T, 3)

        # ── Gated EGNN message passing: batched graph-repetition (EGNO style) ──
        # Replicate the shared Eulerian graph T times with cumulative-offset
        # edge indices, run the GNN ONCE on a (T·BN)-node batched graph instead
        # of looping T times. Same math, 3-5× fewer kernel launches.
        E = edge_index.shape[1]
        device = edge_index.device

        # Flatten the T time slices of h into one batch: (T, BN, C) → (T·BN, C)
        h_batched = h.reshape(T * BN, C)
        # Positions are the same for every time slice (Eulerian mesh): repeat T×.
        x_batched = x.repeat(T, 1)                       # (T·BN, 3)
        # Per-point T-frame velocity history is shared across all T time slices —
        # each GNN "copy" consumes the same vel_all to build edge features.
        vel_all_batched = vel_all_new.repeat(T, 1, 1)    # (T·BN, T, 3)
        # Edges: T copies with cumulative offsets of BN per copy.
        offsets = (torch.arange(T, device=device) * BN).repeat_interleave(E)  # (T·E,)
        edges_batched = edge_index.repeat(1, T) + offsets.unsqueeze(0)        # (2, T·E)

        h_out, _, _ = self.gnn(h_batched, x_batched, vel_all_batched, edges_batched)
        h_new = h_out.reshape(T, BN, C)

        return h_new, vel_all_new


class GatedEGNOModel(ResidualModel):
    """Gated EGNO: stacked (TimeConv_h + TimeConv_v + gated-EGNN) blocks + one-shot decode."""

    FEATURES = ["udf_truncated", "udf_gradient", "knn_graph"]

    # Defaults — overridable via constructor kwargs.
    hidden_dim   = 32
    depth        = 4      # EGNO uses 4-5 blocks by default in their repo
    n_modes      = 3      # FFT modes for length-5 sequences (5//2 + 1 = 3)
    dropout      = 0.0
    no_slip_mask = True
    heads        = 1      # per-edge gate heads (EGNN paper Eq 3.3 is heads=1)
    use_gate     = True   # set False to ablate the per-edge sigmoid gate

    # Per-time-slice scalar input dim: vel_mag(1) + udf_trunc(1) + |udf_grad|(1) = 3.
    per_frame_input_dim = 3

    # `preserves_init` tells src/train.py not to clobber our custom init
    # (same convention as FixedEGNNModel).
    preserves_init = True

    def __init__(
        self,
        features=None,
        depth=None,
        hidden_dim=None,
        dropout=None,
        update_coords=None,
        no_slip_mask=None,
        n_modes=None,
        heads=None,
        use_gate=None,
        **kwargs,
    ):
        if not _SCATTER_AVAILABLE:
            raise ImportError(
                "torch_scatter is required for GatedEGNOModel. "
                "Install via the Modal image (pinned in src/modal_image.py)."
            )
        if not _NEURALOP_AVAILABLE:
            raise ImportError(
                "neuraloperator is required for GatedEGNOModel. "
                "Install with `pip install neuraloperator`."
            )
        if features      is not None: self.FEATURES     = features
        if depth         is not None: self.depth        = depth
        if hidden_dim    is not None: self.hidden_dim   = hidden_dim
        if dropout       is not None: self.dropout      = dropout
        if update_coords is not None: self.update_coords = update_coords
        if no_slip_mask  is not None: self.no_slip_mask = no_slip_mask
        if n_modes       is not None: self.n_modes      = n_modes
        if heads         is not None: self.heads        = heads
        if use_gate      is not None: self.use_gate     = use_gate
        self.update_coords = getattr(self, "update_coords", False)

        super().__init__()

        # Per-frame scalar encoder: 3 scalars → hidden. Shared across time.
        # Concatenate a sinusoidal timestep embedding (hidden_dim//2) so the
        # model can distinguish frame index, matching EGNO's timestep-emb trick.
        self.t_emb_dim = max(8, self.hidden_dim // 2)
        self.per_frame_encoder = nn.Sequential(
            nn.Linear(self.per_frame_input_dim + self.t_emb_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
        )

        # Stacked EGNO blocks.
        self.blocks = nn.ModuleList([
            GatedEGNOBlock(
                self.hidden_dim, n_modes=self.n_modes,
                dropout=self.dropout, update_coords=self.update_coords,
                heads=self.heads,
                use_gate=self.use_gate,
            )
            for _ in range(self.depth)
        ])

        # One-shot decoder: one EquivariantDecoder shared across all T output slices.
        # (EquivariantDecoder outputs Δv per node from (h, m_ij, x, vel_all, edges).)
        self.decoder = EquivariantDecoder(self.hidden_dim)

        # === USUAL INIT ===
        # kaiming_normal_ on every nn.Linear submodule (including gate layers).
        # This overrides the zero-init set by `FixedEGNNGatedLayer.__init__`,
        # so gates start random in (0,1) per edge instead of uniformly 0.5.
        # All validated runs up to 2026-04-18 used this (buggy) init.
        self._init_weights()

        # === FIXED INIT (testing 2026-04-18) ===
        # Restore the gate layers' intentional zero-init so sigmoid(gate) = 0.5
        # uniformly at step 0 — neutral start, matches the docstring intent.
        # To revert to USUAL INIT: comment out this block.
        for block in self.blocks:
            if hasattr(block.gnn, 'gate'):
                nn.init.zeros_(block.gnn.gate.weight)
                if block.gnn.gate.bias is not None:
                    nn.init.zeros_(block.gnn.gate.bias)

    # ---- init ----
    def _init_weights(self):
        depth_scale = 1.0 / math.sqrt(self.depth) if self.depth > 1 else 1.0
        for mod in [self.per_frame_encoder, *self.blocks]:
            for sub in mod.modules():
                if isinstance(sub, nn.Linear):
                    nn.init.kaiming_normal_(sub.weight, a=0, mode='fan_in', nonlinearity='relu')
                    sub.weight.data *= depth_scale
                    if sub.bias is not None:
                        nn.init.zeros_(sub.bias)

    # ---- edge index ----
    def _build_edge_index(self, knn_graph: Tensor) -> Tensor:
        B, N, k = knn_graph.shape
        device = knn_graph.device
        batch_offsets = torch.arange(B, device=device) * N
        rows = torch.arange(N, device=device).view(1, N, 1).expand(B, N, k)
        rows = rows + batch_offsets.view(B, 1, 1)
        cols = knn_graph + batch_offsets.view(B, 1, 1)
        valid = knn_graph >= 0
        return torch.stack([rows[valid], cols[valid]], dim=0)

    # ---- required by ResidualModel.__init__ ----
    def _predict_delta(self, *args, **kwargs):
        raise NotImplementedError("GatedEGNOModel uses forward() directly (one-shot).")

    # ---- forward ----
    def forward(
        self,
        t: Tensor,
        pos: Tensor,                         # (B, N, 3)
        idcs_airfoil: list[Tensor],
        velocity_in: Tensor,                 # (B, T_in, N, 3)
        point_features: Tensor | None = None,   # (B, N, 4): udf_trunc + udf_grad
        knn_graph: Tensor | None = None,        # (B, N, k)
    ) -> Tensor:                              # (B, T_out, N, 3)
        if self.FEATURES and (point_features is None or knn_graph is None):
            cpf, ckg = self._compute_batch_features(pos, idcs_airfoil)
            if point_features is None: point_features = cpf
            if knn_graph      is None: knn_graph      = ckg

        B, T, N, _ = velocity_in.shape
        BN = B * N
        edge_index = self._build_edge_index(knn_graph)
        pos_flat = pos.reshape(BN, 3)

        # ── Per-frame scalar features: [vel_mag, udf_trunc, udf_grad_mag] ──
        vel_mag = velocity_in.norm(dim=-1).transpose(1, 2)       # (B, N, T)
        if point_features is not None:
            udf_t = point_features[..., 0:1]                      # (B, N, 1)
            udf_g = point_features[..., 1:4].norm(dim=-1, keepdim=True)  # (B, N, 1)
        else:
            udf_t = torch.zeros(B, N, 1, device=pos.device, dtype=velocity_in.dtype)
            udf_g = torch.zeros_like(udf_t)

        # Stack into (T, BN, 3) — time becomes leading axis like EGNO.
        vm = vel_mag.unsqueeze(-1)                                # (B, N, T, 1)
        ut = udf_t.unsqueeze(2).expand(-1, -1, T, -1)             # (B, N, T, 1)
        ug = udf_g.unsqueeze(2).expand(-1, -1, T, -1)             # (B, N, T, 1)
        per_frame = torch.cat([vm, ut, ug], dim=-1)               # (B, N, T, 3)
        # Rearrange to (T, B*N, 3)
        per_frame = per_frame.permute(2, 0, 1, 3).reshape(T, BN, 3)

        # Timestep embeddings: (T, t_emb_dim), broadcast to (T, BN, t_emb_dim).
        t_emb = sinusoidal_time_embedding(
            torch.arange(T, device=pos.device), self.t_emb_dim,
        )
        t_emb = t_emb.unsqueeze(1).expand(T, BN, self.t_emb_dim)

        # Encode per-frame (scalars + time embedding) → hidden.
        h_in = torch.cat([per_frame, t_emb], dim=-1)              # (T, BN, 3 + t_emb_dim)
        h = self.per_frame_encoder(h_in.reshape(T * BN, -1)).reshape(T, BN, self.hidden_dim)

        # ── Vector channel: velocity history per point ──
        # (BN, T, 3) — used by both TimeConv_x and the gated GNN layer.
        vel_all = velocity_in.permute(0, 2, 1, 3).reshape(BN, T, 3)

        # ── Stacked EGNO blocks ──
        for block in self.blocks:
            h, vel_all = block(h, pos_flat, vel_all, edge_index)

        # ── Decode all T output slices in one batched graph call ──
        # Replicate graph T× (same trick as inside GatedEGNOBlock) so we run
        # the final GNN + decoder once on the (T·BN)-node graph, not T times.
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
        )                                            # (T·BN, 3)
        delta = delta_flat.reshape(T, B, N, 3).permute(1, 0, 2, 3).contiguous()  # (B, T, N, 3)

        # ── Residual prior: each output frame = last input frame + predicted Δv ──
        last_frame = velocity_in[:, -1:]                            # (B, 1, N, 3)
        v_out = last_frame + delta                                  # (B, T, N, 3)

        # ── No-slip mask at airfoil surface ──
        if self.no_slip_mask:
            vol_mask = torch.ones(B, N, 1, device=pos.device, dtype=velocity_in.dtype)
            for b, idc in enumerate(idcs_airfoil):
                vol_mask[b, idc, :] = 0.0
            v_out = v_out * vol_mask.unsqueeze(1)

        return v_out
