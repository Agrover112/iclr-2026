"""Base class for all GRaM competition models."""

import torch
import torch.nn as nn
from torch import Tensor

from src.features import compute_point_features, total_feature_dim


class ResidualModel(nn.Module):
    """
    Base class for residual velocity prediction models.

    Why residual?
        Model predicts delta = velocity_out - velocity_in[-1] (~1.68 m/s)
        instead of absolute velocity_out (~37.76 m/s). ~22x smaller target
        → easier to learn with only 810 training samples.
        The last input frame is added back inside forward(), so the competition
        always receives absolute velocities. Training loss against absolute
        velocity_out is mathematically equivalent to loss against the residual.

    Subclasses must implement:
        FEATURES: list[str]   — feature names from FEATURE_REGISTRY
        _predict_delta(...)   — network that maps inputs → velocity delta

    The base class handles:
        - Computing point features on the fly (inference) or accepting precomputed
          ones from the dataloader (training, much faster)
        - Adding velocity_in[:, -1:] to delta → absolute velocity output
    """

    # Subclass must declare which features it uses, e.g. ["udf_truncated", "udf_gradient"]
    FEATURES: list[str] = []

    @property
    def feature_dim(self) -> int:
        """Total input channels from point features."""
        return total_feature_dim(self.FEATURES)

    def _predict_delta(
        self,
        t: Tensor,
        pos: Tensor,
        idcs_airfoil: list[Tensor],
        velocity_in: Tensor,
        point_features: Tensor,
    ) -> Tensor:
        """Predict velocity residual (delta).

        Args:
            t:              (batch, 10)
            pos:            (batch, N, 3)
            idcs_airfoil:   list[Tensor] — surface indices per sample
            velocity_in:    (batch, 5, N, 3)
            point_features: (batch, N, F)

        Returns:
            delta: (batch, 5, N, 3) — predicted change from velocity_in[:, -1]
        """
        raise NotImplementedError

    def _compute_batch_features(
        self,
        pos: Tensor,
        idcs_airfoil: list[Tensor],
    ) -> Tensor:
        """Compute point features for a batch on the fly.

        Used at inference time (competition). During training, the dataloader
        provides precomputed features from disk — much faster.

        Args:
            pos:          (batch, N, 3)
            idcs_airfoil: list of length batch

        Returns:
            (batch, N, F) float32
        """
        batch_feats = []
        for b in range(pos.shape[0]):
            surface_pts = pos[b][idcs_airfoil[b]]               # (M, 3)
            feat = compute_point_features(pos[b], surface_pts, self.FEATURES)  # (N, F)
            batch_feats.append(feat)
        return torch.stack(batch_feats)                          # (batch, N, F)

    def forward(
        self,
        t: Tensor,
        pos: Tensor,
        idcs_airfoil: list[Tensor],
        velocity_in: Tensor,
        point_features: Tensor | None = None,
    ) -> Tensor:
        """Forward pass — matches competition interface when called without point_features.

        Training (fast):   model(t, pos, idcs, vel_in, preloaded_features)
        Inference (slow):  model(t, pos, idcs, vel_in)  ← computes features on the fly

        Returns absolute velocity: (batch, 5, N, 3)
        """
        if point_features is None:
            point_features = self._compute_batch_features(pos, idcs_airfoil)

        delta = self._predict_delta(t, pos, idcs_airfoil, velocity_in, point_features)
        return delta + velocity_in[:, -1:]   # residual → absolute
