# GatedEGNOMeanResModel — training process

## Architecture

E(n)-equivariant graph neural operator with spectral temporal mixing
and one-shot 5-frame prediction against a temporal-mean residual reference
(Reynolds decomposition).

Blocks: 4 × (TimeConv on hidden + TimeConvX on velocity + FixedEGNNGatedLayer
message passing). Hidden dim 64, single-scalar edge gate per layer.

## Data

- Split: 70% train / 15% val / 15% test, stratified by simulation key
  `{geometry_id}_{sim_id}` (5 chunks per simulation stay in the same split).
- 162 simulations × 5 chunks = 810 training files total; 570 used for training.
- Features precomputed per sample: truncated UDF (d_max=0.5), UDF gradient,
  fixed-k kNN graph (k=16).

## Training

- Optimizer: AdamW, lr 2e-3, weight decay 0.01
- Schedule: linear warmup (10 epochs) → cosine annealing
- Batch size 1, grad accumulation 8 (effective batch 8)
- Loss: MSE on absolute velocity targets
- 100 epochs max, patience 20 on val metric (L2 norm averaged over
  timesteps, points, and batch)
- Single seed (42); no ensembling at submission time
- GPU: NVIDIA B200 via Modal

## Key training decisions

1. **Mean-residual reference** — predicting `Δv = v_out − mean(v_in)` instead
   of `Δv = v_out − v_in[-1]`. The temporal mean has lower variance than any
   single frame, giving a cleaner regression target in turbulent-wake regions.

2. **FIXED INIT** — the edge-gate layer is explicitly zero-initialized after
   the parent `_init_weights()` call so sigmoid(gate)=0.5 uniformly at step 0.
   This contrasts with letting kaiming_normal override the zero-init, which
   produces random gate values in (0,1) per edge and causes larger initial
   gradient spikes.

3. **No autoregressive rollout** — all 5 output frames are decoded from a
   single forward pass, avoiding the error compounding that a predict-one-step-
   feed-back loop produces on chaotic wake dynamics.

## Features recomputed at inference

The training pipeline caches UDF, UDF gradient, and kNN graph per sample in
`_feat.pt` files. At inference time the submission model recomputes these on
the fly from `(pos, idcs_airfoil)` using the same math (torch.cdist for UDF,
scipy.cKDTree for kNN). Feature definitions are inlined in `model.py`.
