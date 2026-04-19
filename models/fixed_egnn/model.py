"""
FixedEGNN — E(n)-equivariant EGNN for the Eulerian warped-IFW task.

Fixes vs. models/egnn/model.py:
  * Scalar channel h carries only INVARIANT features:
      - per-timestep velocity magnitudes   (5,)
      - udf_truncated                      (1,)
      - |udf_gradient|                     (1,)
      total = 7 scalars
  * All 5 velocity frames enter the message through EDGE-DIRECTION
    PROJECTIONS (vel^(t) · r_hat) — 5 invariant scalars per endpoint,
    10 extra a_ij scalars per edge. φ_e can now learn temporal patterns
    (acceleration, reversal) along each edge direction.
  * Decoder is EQUIVARIANT: Δv = Σ_t α^(t)(h)·v^(t) + geometry term.
    All 5 frames contribute as equivariant basis vectors weighted by
    invariant scalar gates from h.
  * Coordinates are NEVER updated by default (Eulerian regime — the mesh
    is fixed across rollout steps). update_coords=True enables the full
    EGNN coord update for experimentation.
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


class FixedEGNNLayer(nn.Module):
    """Invariant h update; all 5 velocity frames enter as edge-projected scalars."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0, update_coords: bool = False):
        super().__init__()
        self.update_coords = update_coords
        # a_ij: dist2 (1) + vel_proj_src×5 + vel_proj_dst×5 = 11 extra scalars
        self.phi_e = nn.Sequential(
            nn.Linear(2 * hidden_dim + 11, hidden_dim),
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

    def forward(
        self, h: Tensor, x: Tensor, vel_all: Tensor, edge_index: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Paper: Satorras, Hoogeboom & Welling (2021), "E(n) Equivariant GNNs",
        # eqs. (3)-(6). Our src=neighbor j, dst=receiver i (opposite naming to
        # the paper, which uses i = receiver). Sign of `rel` is therefore
        # flipped vs. eq. (4); the learnable scalar phi_x absorbs the sign.
        #
        # ── Symbol mapping (paper → code) ───────────────────────────────
        #   i (receiver)            →  dst  (row in edge_index)
        #   j (sender / neighbor)   →  src  (row in edge_index)
        #   h_i^l                   →  h[dst]          (N, hidden)
        #   h_j^l                   →  h[src]          (N, hidden)
        #   x_i, x_j                →  x[dst], x[src]  (N, 3)  — fixed mesh
        #   ‖x_i − x_j‖²            →  dist2           (E, 1)
        #   (x_i − x_j)/‖·‖         →  -r_hat          (E, 3)  (sign flip: we use x_j−x_i)
        #   a_ij (edge attribute)   →  [dist2, vel_proj_src(5), vel_proj_dst(5)]  (E, 11)
        #   m_ij                    →  m_ij            (E, hidden)
        #   m_i = Σ_{j} m_ij        →  agg             (N, hidden)  — scatter_add over dst
        #   φ_e                     →  self.phi_e      (2·hidden + 11 → hidden)
        #   φ_h                     →  self.phi_h      (2·hidden      → hidden)
        #   φ_x                     →  self.phi_x      (hidden        → 1)   — only used if update_coords=True
        #   C (normalization)       →  scatter_mean    (implicit 1/|N(i)|)
        #
        # Extension vs. paper: we carry all 5 velocity frames as an extra
        # equivariant input `vel_all`; the paper's a_ij is scalar-only, ours
        # includes 10 extra invariant scalars (vel^(t)·r̂ for t=1..5 at each
        # endpoint) so φ_e can see per-frame edge-aligned flow dynamics.
        #
        # vel_all: (B*N, 5, 3) — all 5 input frames, stacked.
        src, dst = edge_index                                   # j, i
        rel   = x[src] - x[dst]                                 # (E, 3)  (x_j - x_i)
        dist2 = (rel * rel).sum(-1, keepdim=True)               # (E, 1)  ‖x_i - x_j‖²
        dist  = dist2.sqrt().clamp(min=1e-8)
        r_hat = rel / dist                                      # (E, 3)  unit edge dir

        # a_ij in eq. (3): 10 invariant scalars — each of the 5 velocity
        # frames projected onto the edge direction at src and dst.
        # Shape: (E, 5) each; captures temporal dynamics (acceleration,
        # reversal) along the edge, not just the instantaneous flow.
        vel_proj_src = (vel_all[src] * r_hat.unsqueeze(1)).sum(-1)  # (E, 5)
        vel_proj_dst = (vel_all[dst] * r_hat.unsqueeze(1)).sum(-1)  # (E, 5)

        # Eq. (3):  m_ij = φ_e( h_i, h_j, ‖x_i − x_j‖², a_ij )
        # a_ij = [dist2, vel_proj_src×5, vel_proj_dst×5]  (11 scalars)
        m_ij = self.phi_e(
            torch.cat([h[src], h[dst], dist2, vel_proj_src, vel_proj_dst], dim=-1)
        )                                                        # (E, hidden)

        # Eq. (4):  x_i ← x_i + Σ_{j≠i} (x_i − x_j) · φ_x(m_ij) / (N−1)
        # Off by default (Eulerian regime). When enabled, coords mutate
        # within the forward pass only; caller resets to fixed mesh each
        # rollout step. Sign flip vs. paper is absorbed by φ_x.
        if self.update_coords:
            coord_w = self.phi_x(m_ij)                                 # (E, 1)
            delta_x = scatter_mean(rel * coord_w, dst, dim=0, dim_size=x.size(0))
            x = x + delta_x

        # Eq. (5):  m_i = Σ_{j≠i} m_ij
        # scatter_add writes each edge message into slot dst[e] (=receiver i),
        # accumulating all incoming messages at i.
        agg = scatter_add(m_ij, dst, dim=0, dim_size=h.size(0))  # (N, hidden)
        # Eq. (6):  h_i ← φ_h( h_i, m_i )   (+ residual + LayerNorm + dropout)
        h = h + self.drop(self.phi_h(torch.cat([h, agg], dim=-1)))
        h = self.norm(h)
        return h, x, m_ij


class EquivariantDecoder(nn.Module):
    """Δv_i = Σ_{t=1}^{5} α_i^(t) · v_i^(t) + Σ_j w(m_ij) · (x_j − x_i).

    Both terms are weighted sums of EQUIVARIANT vectors by INVARIANT
    scalars, so the output is E(n)-equivariant by construction.

    The temporal gates α^(t) let the model express things like
    "output ≈ last frame + correction in the direction of the acceleration
    trend" — all without breaking equivariance.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.edge_weight = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # 5 independent scalar gates, one per input velocity frame.
        self.vel_gates = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 5),   # (N, 5) — one gate per timestep
        )

        # === ACTIVE: fix_v2_linear — PyTorch default init ===
        # Lets gradients flow to all decoder weights from step 0, so UDF /
        # GNN messages actually influence the output. Verified on 2026-04-14:
        # val_metric 1.5117 / test 1.1948 @ 30 epochs, 15% data — beats the
        # naive-init dead-zone baseline (val 1.5712 / test 1.2534) by ~4-5%.
        # Residual prior is NOT enforced (Δv random at init), so the model
        # needs warmup (warmup_epochs=5) and a lower LR (3e-4, not 1e-3) to
        # recover from the initial random explosion. See LOG.md 2026-04-14.
        pass

        # === naive zero-init (DISABLED) — residual prior Δv=0 at init ===
        # To re-enable: delete `pass` above and uncomment the two loops below.
        # Known pathology: dead zone where W1/W2 never get gradient — only the
        # final bias b2 learns. UDF / GNN messages end up ignored entirely.
        # Stable naive-baseline start but caps at ~1.57 val_metric.
        # for m in self.edge_weight.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.zeros_(m.weight)
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)
        # for m in self.vel_gates.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.zeros_(m.weight)
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)

    def forward(
        self,
        h: Tensor,
        m_ij: Tensor,
        x: Tensor,
        vel_all: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        # Geometry term: equivariant weighted sum of relative positions.
        src, dst = edge_index
        rel = x[src] - x[dst]                                           # (E, 3)
        w   = self.edge_weight(m_ij)                                    # (E, 1)
        geom_delta = scatter_mean(rel * w, dst, dim=0, dim_size=h.size(0))  # (N, 3)

        # Temporal term: Σ_t α^(t) · v^(t).
        # vel_all: (N, 5, 3); alpha: (N, 5) → unsqueeze → (N, 5, 1)
        alpha = self.vel_gates(h).unsqueeze(-1)                         # (N, 5, 1)
        vel_combo = (alpha * vel_all).sum(dim=1)                        # (N, 3)

        return vel_combo + geom_delta                                   # (N, 3)


class FixedEGNNModel(ResidualModel):

    # Tells src/train.py not to clobber the decoder's zero-init of
    # edge_weight + vel_gates (which encodes the residual prior
    # Δv = 0 → next_vel = last frame at init).
    preserves_init = True

    FEATURES = ["udf_truncated", "udf_gradient", "knn_graph"]

    hidden_dim    = 32
    depth         = 2
    dropout       = 0.0
    update_coords = False  # Eulerian default; set True for equivariant coord updates
    # No-slip: airfoil surface enforces v=0 at all times. When True, predictions
    # at idcs_airfoil are multiplicatively masked to 0 at every rollout step
    # (mask multiplication, autograd-safe — no in-place ops).
    no_slip_mask  = True

    def __init__(self, features=None, depth=None, hidden_dim=None, dropout=None,
                 update_coords=None, no_slip_mask=None, **kwargs):
        if not _SCATTER_AVAILABLE:
            raise ImportError(
                "torch_scatter is required for FixedEGNNModel. "
                "Install via the Modal image (already pinned in src/modal_image.py)."
            )
        if features      is not None: self.FEATURES      = features
        if depth         is not None: self.depth         = depth
        if hidden_dim    is not None: self.hidden_dim    = hidden_dim
        if dropout       is not None: self.dropout       = dropout
        if update_coords is not None: self.update_coords = update_coords
        if no_slip_mask  is not None: self.no_slip_mask  = no_slip_mask
        super().__init__()

        # Invariant scalar node input: 5 vel mags + udf_truncated (1) + |udf_grad| (1) = 7.
        # local_density slot was tested on 2026-04-14 and made things worse (val 1.95
        # vs 1.68 baseline at ep 10) — see LOG.md. Kept at 7.
        node_input_dim = 7

        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList([
            FixedEGNNLayer(self.hidden_dim, dropout=self.dropout, update_coords=self.update_coords)
            for _ in range(self.depth)
        ])
        self.decoder = EquivariantDecoder(self.hidden_dim)

        self._init_weights()

        weights_path = os.path.join(os.path.dirname(__file__), "state_dict.pt")
        if os.path.exists(weights_path):
            self.load_state_dict(torch.load(weights_path, weights_only=True))

    def _init_weights(self):
        # depth_scale = 1/sqrt(depth) — weight suppression to prevent variance
        # compounding through deep stacks. Tried depth_scale=1.0 for d=6 and it
        # helped convergence speed there, but for d=8 the initial explosion
        # jumped 4.3x (val @ ep1: 52.9 vs 12.4 for d=6). Restoring depth_scale
        # to scale cleanly with deeper models as we go to full data later.
        # See LOG.md 2026-04-14.
        depth_scale = 1.0 / math.sqrt(self.depth) if self.depth > 1 else 1.0
        for mod in [self.node_encoder, *self.layers]:
            for sub in mod.modules():
                if isinstance(sub, nn.Linear):
                    nn.init.kaiming_normal_(sub.weight, a=0, mode='fan_in', nonlinearity='relu')
                    sub.weight.data *= depth_scale
                    if sub.bias is not None:
                        nn.init.zeros_(sub.bias)

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
        vel_window: Tensor,              # (B, 5, N, 3)
        point_features: Tensor | None,   # (B, N, 4): udf_trunc(1) + udf_grad(3)
        edge_index: Tensor,
    ) -> Tensor:
        B, _, N, _ = vel_window.shape

        # ── Invariant scalar node features ──────────────────────────────
        vel_mag = vel_window.norm(dim=-1).transpose(1, 2)          # (B, N, 5)
        if point_features is not None:
            udf_trunc    = point_features[..., 0:1]                # (B, N, 1)
            udf_grad_mag = point_features[..., 1:4].norm(dim=-1, keepdim=True)  # (B, N, 1)
            scalars = torch.cat([vel_mag, udf_trunc, udf_grad_mag], dim=-1)     # (B, N, 7)
        else:
            zeros = torch.zeros(B, N, 2, device=vel_window.device, dtype=vel_window.dtype)
            scalars = torch.cat([vel_mag, zeros], dim=-1)

        h = self.node_encoder(scalars.reshape(B * N, -1))           # (B*N, hidden)

        # ── Equivariant vector channel: all 5 frames ─────────────────────
        # Reshape to (B*N, 5, 3) so layers and decoder can project per-frame.
        vel_all_flat = vel_window.permute(0, 2, 1, 3).reshape(B * N, 5, 3)

        # x mutates within the pass only when update_coords=True;
        # pos_flat is never written to — next rollout step starts fresh.
        x = pos_flat.clone() if self.update_coords else pos_flat
        m_ij_last = None
        for layer in self.layers:
            h, x, m_ij_last = layer(h, x, vel_all_flat, edge_index)

        delta = self.decoder(h, m_ij_last, x, vel_all_flat, edge_index)
        return delta.reshape(B, N, 3)

    def _predict_delta(self, *args, **kwargs):
        raise NotImplementedError("FixedEGNNModel uses forward() directly.")

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
        pos_flat = pos.reshape(B * N, 3)  # Eulerian — never mutated

        # No-slip volume mask: 1 everywhere, 0 at airfoil indices. Precomputed
        # once per forward since idcs_airfoil is constant across rollout steps.
        # Multiplicative (autograd-safe, no in-place). Applied inside the loop
        # so clamped values feed back into vel_mag / vel_proj for the next step.
        if self.no_slip_mask:
            vol_mask = torch.ones(B, N, 1, device=pos.device, dtype=velocity_in.dtype)
            for b, idc in enumerate(idcs_airfoil):
                vol_mask[b, idc, :] = 0.0
        else:
            vol_mask = None

        window = velocity_in.clone()
        preds = []
        for _ in range(5):
            delta    = self._egnn_step(pos_flat, window, point_features, edge_index)
            next_vel = window[:, -1] + delta
            if vol_mask is not None:
                next_vel = next_vel * vol_mask
            preds.append(next_vel.unsqueeze(1))
            window = torch.cat([window[:, 1:], next_vel.unsqueeze(1)], dim=1)
        return torch.cat(preds, dim=1)