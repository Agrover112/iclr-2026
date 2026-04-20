# Training process

## Overview

This submission predicts the next five frames of a 3D velocity field on
an irregular 100k-point cloud surrounding one to three airfoils. The
task has three invariances that motivate the architecture:

- Rotation and translation of the whole point cloud leave the physics
  unchanged (the Navier–Stokes equations are E(3)-equivariant).
- The airfoil surface imposes a no-slip boundary: v = 0 on all points
  in `idcs_airfoil`.
- The flow is dominated by advection with a predictable mean component
  and a smaller, high-frequency turbulent fluctuation, making it natural
  to decompose the target via Reynolds' mean–fluctuation split.

The model respects all three by construction.

## Architecture

The backbone is an E(n)-equivariant graph neural operator in the style
of EGNO (Xu et al., 2024; arXiv:2401.11037), built on top of the
Satorras–Hoogeboom–Welling EGNN layer (2021; arXiv:2102.09844). Each
of the four stacked blocks performs, in order:

1. a 1-D spectral convolution along the input-time axis on the scalar
   hidden channel (`TimeConv`);
2. the same spectral convolution applied per-component on the raw
   velocity vectors, treating the three spatial coordinates as a
   preserved batch dimension so that rotation equivariance is maintained
   (`TimeConvX`); and
3. an EGNN message-passing step on a fixed 16-nearest-neighbour graph,
   with a sigmoid edge gate (equivalent to the edge-inference mechanism
   in EGNN Eq. 3.3).

Temporal spectral convolutions use the `neuraloperator` package's
`SpectralConv` (Li et al., 2021; arXiv:2010.08895). For a five-frame
input window, the maximum usable number of Fourier modes is
`T // 2 + 1 = 3`, so all modes are retained.

A final equivariant decoder produces the five output velocity
increments from each block's final hidden state as
`Δv_i = Σ_t α_i^(t) v_i^(t) + Σ_j w(m_ij) (x_j − x_i)`, a weighted
combination of the equivariant input frames and relative positions
with invariant scalar weights — equivariant by construction.

The five output frames are decoded in **one shot** rather than
autoregressively. This avoids the exposure-bias compounding that makes
autoregressive GNN rollouts drift on chaotic wake dynamics.

## Input features

Per-point invariant scalars are derived from the geometry:

- `udf_truncated`: distance to the nearest airfoil-surface point,
  clamped at `d_max = 0.5`. Truncation focuses numerical resolution on
  the boundary layer.
- `udf_gradient`: unit vector toward the nearest surface point
  (equivariant; enters the network through its magnitude or via
  edge-projection).
- A fixed k-nearest-neighbour graph with `k = 16`.

No features are cached; they are computed on the fly inside `forward`
from `(pos, idcs_airfoil)` and therefore require no side files.

## Target and loss

The network predicts the temporal-mean residual

    Δv = v_out − (1/T) Σ_t v_in[t],

and the output velocity is reconstructed as `v_out = mean(v_in) + Δv`.
This Reynolds decomposition (Reynolds 1895) separates the slowly
varying mean flow — the easy, predictable part — from the chaotic
fluctuations that dominate the prediction error. Because the mean has
lower variance than any single input frame, using it as the reference
gives the regressor a cleaner, lower-variance target. The loss is
ordinary MSE on the reconstructed `v_out`.

## Training

- Optimizer: AdamW, learning rate 2 · 10⁻³, weight decay 10⁻².
- Schedule: 15-epoch linear warmup followed by cosine annealing.
- Batch size 1, gradient accumulation 8 (effective batch 8).
- Maximum 120 epochs, early stopping with patience 30 on the
  validation L2 metric.
- Data split: 154 simulations for training, 4 for validation, 4 for
  test, stratified by `{geometry_id}_{sim_id}` so that all five
  temporal chunks of a simulation stay inside one split. This gives
  770 training files out of 810 total.
- Single seed (42). No ensembling.

Timestep indices 0…T−1 are supplied to the encoder as a sinusoidal
positional embedding (Vaswani et al., 2017; arXiv:1706.03762),
concatenated with the invariant node scalars before the shared MLP.

## Hardware and wall time

Architectural ablations (hidden-dim sweep, residual-reference choice,
gating variants, number of attention heads) were carried out on 15% of
the training data using NVIDIA L40S GPUs (48 GB) on Modal, with
30-epoch budgets that kept each experiment under USD 10. Once the
final configuration was fixed, the submission model was retrained on
the full split above using a single NVIDIA B200 (180 GB HBM3), also
provisioned via Modal. The submission run converged in approximately
seven wall-clock hours at a cost of roughly USD 45.

## Acknowledgements

We gratefully acknowledge Modal for the GPU credits that made this
work possible.
