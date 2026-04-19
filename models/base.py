"""Base class for all GRaM competition models."""

import torch
import torch.nn as nn
from torch import Tensor

from src.features import compute_point_features, total_feature_dim, FEATURE_REGISTRY, GRAPH_FEATURES
from scipy.spatial import cKDTree


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
        - Computing point features + knn_graph on the fly (inference) or
          accepting precomputed ones from the dataloader (training, much faster)
        - Adding velocity_in[:, -1:] to delta → absolute velocity output
    """

    # Subclass must declare which features it uses
    # e.g. ["udf_truncated", "udf_gradient", "knn_graph"]
    FEATURES: list[str] = []

    KNN_K: int = 16  # k for on-the-fly uniform knn_graph construction at inference

    @property
    def feature_dim(self) -> int:
        """Total input channels from float point features (excludes graph index features)."""
        return total_feature_dim([f for f in self.FEATURES if f not in GRAPH_FEATURES])

    def _predict_delta(
        self,
        t: Tensor,
        pos: Tensor,
        idcs_airfoil: list[Tensor],
        velocity_in: Tensor,
        point_features: Tensor | None,
        knn_graph: Tensor | None,
    ) -> Tensor:
        """Predict velocity residual (delta).

        Args:
            t:              (batch, 10)
            pos:            (batch, N, 3)
            idcs_airfoil:   list[Tensor] — surface indices per sample
            velocity_in:    (batch, 5, N, 3)
            point_features: (batch, N, F) float features (UDF etc.), or None
            knn_graph:      (batch, N, k) int64 neighbor indices, or None

        Returns:
            delta: (batch, 5, N, 3) — predicted change from velocity_in[:, -1]
        """
        raise NotImplementedError

    def _compute_batch_features(
        self,
        pos: Tensor,
        idcs_airfoil: list[Tensor],
    ) -> tuple[Tensor | None, Tensor | None]:
        """Compute point features + knn_graph on the fly for a batch.

        Used at inference time (competition). During training, the dataloader
        provides precomputed features from disk — much faster.

        Args:
            pos:          (batch, N, 3)
            idcs_airfoil: list of length batch

        Returns:
            point_features: (batch, N, F) float32, or None if no float features
            knn_graph:      (batch, N, k) int64, or None if not in FEATURES
        """
        float_features = [f for f in self.FEATURES if f not in GRAPH_FEATURES]
        graph_feature  = next((f for f in self.FEATURES if f in GRAPH_FEATURES), None)

        batch_feats = []
        batch_knn = []

        for b in range(pos.shape[0]):
            surface_pts = pos[b][idcs_airfoil[b]]  # (M, 3)

            if float_features:
                feat = compute_point_features(pos[b], surface_pts, float_features)
                batch_feats.append(feat)

            if graph_feature is not None:
                # Compute the requested graph on the fly using the feature registry
                knn = FEATURE_REGISTRY[graph_feature](pos[b], surface_pts)
                batch_knn.append(knn)  # (N, k) or (N, k_near) with -1 padding

        point_features = torch.stack(batch_feats) if batch_feats else None
        knn_graph = torch.stack(batch_knn) if batch_knn else None

        return point_features, knn_graph

    def forward(
        self,
        t: Tensor,
        pos: Tensor,
        idcs_airfoil: list[Tensor],
        velocity_in: Tensor,
        point_features: Tensor | None = None,
        knn_graph: Tensor | None = None,
    ) -> Tensor:
        """Forward pass — matches competition interface when called without features.

        Training (fast):   model(t, pos, idcs, vel_in, point_features, knn_graph)
        Inference (slow):  model(t, pos, idcs, vel_in)  ← computes on the fly

        Returns absolute velocity: (batch, 5, N, 3)
        """
        if point_features is None and knn_graph is None and self.FEATURES:
            point_features, knn_graph = self._compute_batch_features(pos, idcs_airfoil)

        delta = self._predict_delta(t, pos, idcs_airfoil, velocity_in, point_features, knn_graph)
        return delta + velocity_in[:, -1:]   # residual → absolute
