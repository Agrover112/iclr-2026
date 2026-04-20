# Training process

## Overview

This submission predicts the next five frames of a 3D velocity field on
an irregular 100k-point cloud surrounding one to three airfoils. Three
structural properties of the task motivate the architecture:

- **$E(3)$-equivariance.** Rotating or translating the input point
  cloud rotates or translates the output velocity field accordingly.
  The Navier–Stokes equations are $E(3)$-equivariant, so their
  solutions — velocity fields — transform as vectors under the group
  action.
- **No-slip boundary condition.** The airfoil surface enforces
  $v = 0$ on all points in `idcs_airfoil`, at every timestep, for
  every geometry.
- **Reynolds mean–fluctuation structure.** The flow is dominated by a
  slowly-varying advective mean with a smaller, high-frequency
  turbulent component — a natural target for mean/fluctuation
  decomposition.

The model respects the first two by construction (equivariant message
passing + a mask that zeros the prediction at `idcs_airfoil`) and
leverages the third through a temporal-mean residual target.

Prediction error on this dataset is heavily concentrated in the
near-wake and boundary layer, where Kelvin–Helmholtz instabilities
and vortex shedding produce high-frequency velocity fluctuations.
Far-field points are nearly stationary and decay in error rapidly
with distance to the airfoil. This motivates the use of spectral
temporal mixing (below) — modelling the flow directly in frequency
space along the time axis is well-suited to the dominant wake physics.

## Architecture

The backbone is an $E(n)$-equivariant graph neural operator in the style
of EGNO (Xu et al., 2024; arXiv:2401.11037), built on top of the
Satorras–Hoogeboom–Welling EGNN layer (2021; arXiv:2102.09844). The
model stacks four blocks of hidden width 96; each block runs three
components in sequence, followed by a shared decoder:

- **`TimeConv`** — a 1-D spectral convolution along the input-time
  axis applied to the scalar hidden channel. Implemented with the
  `neuraloperator` package's **`SpectralConv`** (Li et al., 2021;
  arXiv:2010.08895). Let $\mathcal{F}_t$ be the FFT along the time
  axis, $\hat{h}(k) = \mathcal{F}_t(h)(k)$ the $k$-th Fourier
  coefficient, $\mathcal{K} = \{0, 1, 2\}$ the retained modes (the
  maximum for $T = 5$, since $\lfloor T/2 \rfloor + 1 = 3$), and
  $W^{(k)}$ the learnable per-mode weight. The update is

$$
h' = h + \mathrm{LeakyReLU}\Bigl( \mathcal{F}_t^{-1}\bigl( W^{(k)} \hat{h}(k) \bigr) \Bigr), \qquad k \in \mathcal{K}
$$

- **`TimeConvX`** — the same spectral convolution applied
  component-wise to the raw velocity, treating the three spatial
  coordinates $d \in \{1,2,3\}$ as a preserved batch dimension so
  that rotation equivariance is maintained:

  With $\hat{v}_d(x_i, k)$ denoting the temporal Fourier coefficient of
  $v_d(x_i, \cdot)$ at mode $k$, and $\tilde W^{(k)}$ the learnable
  per-mode weight,

$$
v'_{d}(x_i, t) = v_{d}(x_i, t) + \mathcal{F}_t^{-1}\Bigl( \tilde W^{(k)} \hat{v}_d(x_i, k) \Bigr)(t), \qquad k \in \mathcal{K}
$$

  Because the same scalar weights $\tilde W^{(k)}$ are applied to each
  spatial component independently, a rotation $R \in O(3)$ of the
  input commutes with this map.

- **`FixedEGNNGatedLayer`** — an EGNN message-passing step on a
  fixed 16-nearest-neighbour graph, with the edge-inference mechanism
  from Section 3.2 of the EGNN paper (here used with a single scalar
  gate per edge, i.e. one gate head). For each edge
  $(i, j) \in \mathcal{E}$, with attributes $a_{ij}$ encoding the
  relative geometry and per-endpoint velocity projections, the block
  runs

$$
\begin{aligned}
m_{ij} &= \phi_e\bigl(h_i^{l}, h_j^{l}, \lVert x_i - x_j \rVert^2, a_{ij}\bigr) && \text{(Eq. 3)} \\\\
e_{ij} &= \phi_{\mathrm{inf}}(m_{ij}) = \sigma\bigl(\mathrm{Linear}(m_{ij})\bigr) && \text{(Eq. 8)} \\\\
m_i &= \sum_{j \in \mathcal{N}(i)} e_{ij} m_{ij} && \text{(Eq. 7)} \\\\
h_i^{l+1} &= \mathrm{LayerNorm}\bigl(h_i^{l} + \phi_h(h_i^{l}, m_i)\bigr) && \text{(Eq. 6)}
\end{aligned}
$$

  The MLP $\phi_{\mathrm{inf}}$ is what the paper calls the *inferring
  edges* network: originally introduced as a soft indicator of whether
  an edge should contribute to message passing. Here we keep the k-NN
  topology fixed and instead use $e_{ij}$ as a learned per-edge
  **gate** — the same arithmetic, interpreted as a per-edge
  weighting over existing edges rather than edge selection.

- **`EquivariantDecoder`** — shared across the four blocks, produces
  the per-point velocity increment from the final hidden state as

$$
\Delta v_i = \sum_{t} \alpha_i^{(t)} v_i^{(t)} + \sum_{j} w(m_{ij})(x_j - x_i)
$$

  a weighted combination of the equivariant input frames and relative
  positions with invariant scalar weights — equivariant by construction.

Every sub-layer above (`TimeConv`, `TimeConvX`, and the node-update
path of `FixedEGNNGatedLayer`) is wrapped in a residual connection,
visible in the $h + (\ldots)$ / $v + (\ldots)$ form of each equation.
Residuals are essential for stable optimisation of the stacked
four-block network.

The five output frames are decoded in **one shot** rather than
autoregressively. This avoids the exposure-bias compounding that makes
autoregressive GNN rollouts drift on chaotic wake dynamics.

## Input features

Let $\mathcal{I}$ denote the set of airfoil-surface indices (the
`idcs_airfoil` tensor) and $\mathcal{S} = \{x_j : j \in \mathcal{I}\}$
the corresponding surface points. For every point $x_i$, let

$$
s^{\star}(x_i) = \arg\min_{s \in \mathcal{S}} \| x_i - s \|_2
$$

denote its nearest surface point. The per-point geometry features are
then:

- **`udf_truncated`** — truncated unsigned distance to the nearest
  surface point,

$$
\tilde{d}(x_i) = \min\bigl( \lVert x_i - s^{\star}(x_i) \rVert_2, d_{\max} \bigr), \qquad d_{\max} = 0.5
$$

  Truncation focuses numerical resolution on the boundary layer while
  saturating the far-field to a constant.

- **`udf_gradient`** — unit vector pointing from $x_i$ toward its
  nearest surface point,

$$
\hat{n}(x_i) = \frac{s^{\star}(x_i) - x_i}{\lVert s^{\star}(x_i) - x_i \rVert_2}
$$

  This is $E(n)$-equivariant. In the current encoder the vector enters
  only through its (constant-1) Euclidean norm $\|\hat{n}(x_i)\|_2$,
  so the directional information is available to future variants but
  is not actively exploited by this submission.

- **k-nearest-neighbour graph** — directed edge set built from Euclidean
  distance in $\mathbb{R}^3$ with $k = 16$,

$$
\mathcal{E}(x_i) = \bigl\{ (i, j) : x_j \in \mathrm{kNN}_k(x_i) \bigr\}, \qquad k = 16
$$

All three are computed inside `forward` from `(pos, idcs_airfoil)`.

## Target and loss

The network predicts the temporal-mean residual

$$
\Delta v(x_i, t) = v_{\mathrm{out}}(x_i, t) - \frac{1}{T} \sum_{t'=0}^{T-1} v_{\mathrm{in}}(x_i, t'),
$$

and the output velocity is reconstructed as

$$
v_{\mathrm{out}}(x_i, t) = \frac{1}{T} \sum_{t'=0}^{T-1} v_{\mathrm{in}}(x_i, t') + \Delta v(x_i, t).
$$

This Reynolds decomposition (Reynolds 1895) separates the slowly
varying mean flow — the easy, predictable part — from the chaotic
fluctuations that dominate the prediction error. Because the mean has
lower variance than any single input frame, using it as the reference
gives the regressor a cleaner, lower-variance target.

The training loss is ordinary mean-squared error on the reconstructed
output,

$$ \mathcal{E}(x_i) = \lbrace (i, j) : x_j \in \mathrm{kNN}_k(x_i) \rbrace, \qquad k = 16$$

where $\hat{v}_{\mathrm{out}}$ is the model's prediction and
$v_{\mathrm{out}}$ the ground-truth future velocity field.

## Training

- **Optimizer:** AdamW, learning rate $2 \cdot 10^{-3}$, weight decay $10^{-2}$.
- **Schedule:** 15-epoch linear warmup followed by cosine annealing.
- **Batch size:** 1, with gradient accumulation 8 (effective batch 8).
- **Epoch budget:** maximum 120 epochs, early stopping with patience 30
  on the validation L2 metric.
- **Data split:** 154 simulations for training, 4 for validation, 4 for
  test, stratified by `{geometry_id}_{sim_id}` so that all five
  temporal chunks of a simulation stay inside one split. This gives
  770 training files out of 810 total.
- **Seed:** 42. No ensembling.

Timestep indices $0, \ldots, T-1$ are supplied to the encoder as a
sinusoidal positional embedding (Vaswani et al., 2017;
arXiv:1706.03762), concatenated with the invariant node scalars
before the shared MLP.

## Hardware and wall time

Architectural ablations (hidden-dim sweep, residual-reference choice,
gating variants, number of gate heads) were carried out on 15% of
the training data using NVIDIA L40S GPUs (48 GB) on Modal, with
30-epoch budgets that kept each experiment under USD 10. Once the
final configuration was fixed, the submission model was retrained on
the full split above using a single NVIDIA B200 (180 GB HBM3), also
provisioned via Modal. The submission run converged in approximately
seven wall-clock hours at a cost of roughly USD 45.

## Acknowledgements

We gratefully acknowledge Modal for the GPU credits that made this
work possible.

## References

1. Reynolds, O. (1895). "On the Dynamical Theory of Incompressible
   Viscous Fluids and the Determination of the Criterion."
   *Philosophical Transactions of the Royal Society of London A.*
   Classical mean/fluctuation decomposition of turbulent flow; the
   source of the residual target used here.

2. Vaswani, A. et al. (2017). "Attention Is All You Need."
   *Advances in Neural Information Processing Systems (NeurIPS).*
   arXiv:1706.03762. Source of the sinusoidal positional embedding
   used for the frame-index encoding.

3. Sanchez-Gonzalez, A., Godwin, J., Pfaff, T., Ying, R., Leskovec,
   J., Battaglia, P. (2020). "Learning to Simulate Complex Physics
   with Graph Networks." *ICML.* arXiv:2002.09405. Prior art on
   particle-based physical simulation with graph networks.

4. Pfaff, T., Fortunato, M., Sanchez-Gonzalez, A., Battaglia, P.
   (2021). "Learning Mesh-Based Simulation with Graph Networks."
   *ICLR.* arXiv:2010.03409. Prior art on mesh-based GNN simulation;
   the encode-process-decode paradigm informed our block structure.

5. Satorras, V. G., Hoogeboom, E., Welling, M. (2021). "E(n)
   Equivariant Graph Neural Networks." *ICML.* arXiv:2102.09844.
   Base message-passing layer and the edge-inference mechanism
   reused for the per-edge gate.

6. Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya,
   K., Stuart, A., Anandkumar, A. (2021). "Fourier Neural Operator
   for Parametric Partial Differential Equations." *ICLR.*
   arXiv:2010.08895. Spectral convolution used for temporal mixing
   inside each block.

7. Xu, M., Han, J., Lou, A., Kossaifi, J., Ramanathan, A.,
   Azizzadenesheli, K., Leskovec, J., Ermon, S., Anandkumar, A.
   (2024). "Equivariant Graph Neural Operator for Modeling 3D
   Dynamics." *ICML.* arXiv:2401.11037. Block structure (spectral
   temporal mixing stacked with equivariant message passing) and
   the overall EGNO framing.
