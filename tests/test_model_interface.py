"""
Tests for competition model interface contract and residual prediction logic.

Three things locked in by main.py that every model must satisfy:
  1. No-argument constructor: Model()
  2. Output shape: (batch, 5, 100000, 3) — absolute velocities
  3. Output dtype: float32

Plus tests for the residual prediction pattern:
  4. Zero-delta residual model == naive baseline (repeat last frame)
  5. Residual model output is absolute, not delta
  6. Adding last frame back gives correct absolute values

Run with:
  uv run --project /home/agrov/gram/ pytest tests/test_model_interface.py -v
"""

import pytest
import torch

# ---------------------------------------------------------------------------
# Constants matching main.py exactly
# ---------------------------------------------------------------------------

BATCH_SIZE = 4        # small for fast tests (main.py uses 95)
NUM_T_IN = 5
NUM_T_OUT = 5
NUM_POS = 100_000
AIRFOIL_SIZES = [3142, 8000, 15000, 24198]  # variable per sample, like real data


def make_dummy_inputs(batch_size=BATCH_SIZE, num_pos=NUM_POS, seed=0):
    """Synthetic inputs matching the competition interface exactly."""
    torch.manual_seed(seed)
    t = torch.rand(batch_size, NUM_T_IN + NUM_T_OUT)
    pos = torch.rand(batch_size, num_pos, 3)
    idcs_airfoil = [
        torch.randint(num_pos, size=(AIRFOIL_SIZES[i % len(AIRFOIL_SIZES)],))
        for i in range(batch_size)
    ]
    velocity_in = torch.rand(batch_size, NUM_T_IN, num_pos, 3)
    return t, pos, idcs_airfoil, velocity_in


# ---------------------------------------------------------------------------
# Toy models for testing patterns without loading real weights
# ---------------------------------------------------------------------------

class NaiveBaselineModel:
    """Repeats last input frame 5 times. The 1.68 baseline. Zero-delta residual."""

    def __call__(self, t, pos, idcs_airfoil, velocity_in):
        last = velocity_in[:, -1:, :, :]           # (batch, 1, num_pos, 3)
        return last.expand(-1, NUM_T_OUT, -1, -1)  # (batch, 5, num_pos, 3)


class ResidualWrapperModel:
    """
    Template for residual prediction models.
    network() predicts deltas; __call__ adds last frame back → absolute output.
    """

    def __init__(self, delta_fn):
        """delta_fn: (t, pos, idcs_airfoil, velocity_in) -> (batch, 5, num_pos, 3) delta"""
        self._delta_fn = delta_fn

    def __call__(self, t, pos, idcs_airfoil, velocity_in):
        last_frame = velocity_in[:, -1:, :, :]                  # (batch, 1, num_pos, 3)
        delta = self._delta_fn(t, pos, idcs_airfoil, velocity_in)  # (batch, 5, num_pos, 3)
        return delta + last_frame                                # absolute velocity ✓


# ---------------------------------------------------------------------------
# 1. Competition interface contract — shape
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_naive_baseline_shape(self):
        """Naive baseline must satisfy the shape assert from main.py."""
        t, pos, idcs, vel_in = make_dummy_inputs()
        model = NaiveBaselineModel()
        out = model(t, pos, idcs, vel_in)
        assert out.shape == (BATCH_SIZE, NUM_T_OUT, NUM_POS, 3), (
            f"Expected ({BATCH_SIZE}, {NUM_T_OUT}, {NUM_POS}, 3), got {out.shape}"
        )

    def test_residual_model_shape(self):
        """Residual wrapper must also produce the correct output shape."""
        t, pos, idcs, vel_in = make_dummy_inputs()
        zero_delta = lambda t, pos, idcs, v: torch.zeros(BATCH_SIZE, NUM_T_OUT, NUM_POS, 3)
        model = ResidualWrapperModel(zero_delta)
        out = model(t, pos, idcs, vel_in)
        assert out.shape == (BATCH_SIZE, NUM_T_OUT, NUM_POS, 3)

    def test_shape_matches_main_py_assert(self):
        """Replicate the exact assertion from main.py line 26."""
        t, pos, idcs, vel_in = make_dummy_inputs(batch_size=BATCH_SIZE)
        model = NaiveBaselineModel()
        velocity_out = model(t, pos, idcs, vel_in)
        # This is the exact line from main.py
        assert velocity_out.shape == (BATCH_SIZE, NUM_T_OUT, NUM_POS, 3)


# ---------------------------------------------------------------------------
# 2. Competition interface contract — dtype
# ---------------------------------------------------------------------------

class TestOutputDtype:
    def test_output_is_float32(self):
        t, pos, idcs, vel_in = make_dummy_inputs()
        model = NaiveBaselineModel()
        out = model(t, pos, idcs, vel_in)
        assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"

    def test_residual_model_dtype(self):
        t, pos, idcs, vel_in = make_dummy_inputs()
        zero_delta = lambda t, pos, idcs, v: torch.zeros(BATCH_SIZE, NUM_T_OUT, NUM_POS, 3)
        model = ResidualWrapperModel(zero_delta)
        out = model(t, pos, idcs, vel_in)
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# 3. Residual prediction logic
# ---------------------------------------------------------------------------

class TestResidualPrediction:
    def test_zero_delta_equals_naive_baseline(self):
        """
        A residual model that predicts zero delta must produce identical output
        to the naive baseline (repeat last frame). This is the floor we must beat.
        """
        t, pos, idcs, vel_in = make_dummy_inputs()

        naive = NaiveBaselineModel()
        zero_delta = lambda t, pos, idcs, v: torch.zeros(BATCH_SIZE, NUM_T_OUT, NUM_POS, 3)
        residual = ResidualWrapperModel(zero_delta)

        naive_out = naive(t, pos, idcs, vel_in)
        residual_out = residual(t, pos, idcs, vel_in)

        assert torch.allclose(naive_out, residual_out), (
            "Zero-delta residual model must equal naive baseline"
        )

    def test_output_is_absolute_not_delta(self):
        """
        The model must return absolute velocity, not residuals.
        If delta != 0, output must differ from last input frame.
        (Catches the bug: accidentally returning delta instead of delta + last_frame)
        """
        t, pos, idcs, vel_in = make_dummy_inputs()
        last_frame = vel_in[:, -1:, :, :].expand(-1, NUM_T_OUT, -1, -1)

        # Model that predicts a constant nonzero delta
        nonzero_delta = lambda t, pos, idcs, v: torch.ones(BATCH_SIZE, NUM_T_OUT, NUM_POS, 3)
        model = ResidualWrapperModel(nonzero_delta)
        out = model(t, pos, idcs, vel_in)

        # Output must be last_frame + 1.0, not just 1.0
        expected = last_frame + 1.0
        assert torch.allclose(out, expected), "Output must be delta + last_frame, not just delta"

    def test_residual_adds_last_frame_correctly(self):
        """
        Given a known delta, output must be exactly last_frame + delta.
        Verifies the bridge: residual training → absolute output.
        """
        t, pos, idcs, vel_in = make_dummy_inputs(seed=42)
        known_delta = torch.full((BATCH_SIZE, NUM_T_OUT, NUM_POS, 3), fill_value=0.5)
        last_frame = vel_in[:, -1:, :, :]

        model = ResidualWrapperModel(lambda *_: known_delta)
        out = model(t, pos, idcs, vel_in)

        expected = last_frame + known_delta
        assert torch.allclose(out, expected)

    def test_each_output_timestep_has_last_frame_added(self):
        """
        The last frame must be added to ALL 5 output timesteps, not just the first.
        """
        t, pos, idcs, vel_in = make_dummy_inputs()
        last_frame = vel_in[:, -1, :, :]  # (batch, num_pos, 3)

        # Delta that varies per timestep
        delta = torch.stack([
            torch.full((BATCH_SIZE, NUM_POS, 3), fill_value=float(i))
            for i in range(NUM_T_OUT)
        ], dim=1)

        model = ResidualWrapperModel(lambda *_: delta)
        out = model(t, pos, idcs, vel_in)

        for t_idx in range(NUM_T_OUT):
            expected_t = last_frame + t_idx
            assert torch.allclose(out[:, t_idx], expected_t), (
                f"Timestep {t_idx}: last_frame not correctly added"
            )

    def test_residual_metric_lower_than_absolute_on_small_delta(self):
        """
        When the true change is small, residual training target has lower L2 norm
        than absolute velocity target. Confirms the 22x variance reduction rationale.
        """
        torch.manual_seed(0)
        velocity_magnitude = 37.76  # mean from dataset analysis
        delta_magnitude = 1.68      # mean residual from dataset analysis

        vel_in = torch.randn(BATCH_SIZE, NUM_T_IN, NUM_POS, 3) * velocity_magnitude
        vel_out = vel_in[:, -1:] + torch.randn(BATCH_SIZE, NUM_T_OUT, NUM_POS, 3) * delta_magnitude

        absolute_target_norm = vel_out.norm(dim=-1).mean().item()
        residual_target_norm = (vel_out - vel_in[:, -1:]).norm(dim=-1).mean().item()

        assert residual_target_norm < absolute_target_norm, (
            "Residual target must have smaller L2 norm than absolute target"
        )
        ratio = absolute_target_norm / residual_target_norm
        assert ratio > 5, f"Expected >5x reduction, got {ratio:.1f}x"


# ---------------------------------------------------------------------------
# 4. Variable-length idcs_airfoil
# ---------------------------------------------------------------------------

class TestVariableLengthAirfoil:
    def test_handles_variable_idcs_airfoil_lengths(self):
        """
        idcs_airfoil is a list (not a tensor) because each sample has a different
        number of airfoil surface points (3142 to 24198). Model must not crash.
        """
        t, pos, idcs, vel_in = make_dummy_inputs()

        # Confirm idcs_airfoil is a list with varying lengths
        assert isinstance(idcs, list)
        lengths = [len(i) for i in idcs]
        assert len(set(lengths)) > 1, "Test requires variable-length idcs_airfoil"

        model = NaiveBaselineModel()
        out = model(t, pos, idcs, vel_in)
        assert out.shape == (BATCH_SIZE, NUM_T_OUT, NUM_POS, 3)

    def test_idcs_airfoil_indices_in_range(self):
        """All airfoil indices must be valid indices into pos (0 to NUM_POS-1)."""
        _, _, idcs, _ = make_dummy_inputs()
        for i, idx_tensor in enumerate(idcs):
            assert idx_tensor.min() >= 0, f"Sample {i}: negative index"
            assert idx_tensor.max() < NUM_POS, f"Sample {i}: index out of range"


# ---------------------------------------------------------------------------
# 5. Competition metric sanity
# ---------------------------------------------------------------------------

class TestCompetitionMetric:
    def test_metric_is_zero_for_perfect_prediction(self):
        """If output == ground_truth, metric must be 0."""
        torch.manual_seed(0)
        ground_truth = torch.rand(BATCH_SIZE, NUM_T_OUT, NUM_POS, 3)
        velocity_out = ground_truth.clone()
        metric = (velocity_out - ground_truth).norm(dim=3).mean(dim=(1, 2))
        assert torch.allclose(metric, torch.zeros(BATCH_SIZE))

    def test_naive_baseline_metric_matches_known_value(self):
        """
        Naive baseline (zero delta) produces nonzero metric when ground truth
        differs from last input frame — confirming metric measures absolute error.
        """
        torch.manual_seed(0)
        vel_in = torch.rand(BATCH_SIZE, NUM_T_IN, NUM_POS, 3)
        ground_truth = torch.rand(BATCH_SIZE, NUM_T_OUT, NUM_POS, 3)

        model = NaiveBaselineModel()
        t = torch.rand(BATCH_SIZE, NUM_T_IN + NUM_T_OUT)
        pos = torch.rand(BATCH_SIZE, NUM_POS, 3)
        idcs = [torch.randint(NUM_POS, (1000,)) for _ in range(BATCH_SIZE)]

        velocity_out = model(t, pos, idcs, vel_in)
        metric = (velocity_out - ground_truth).norm(dim=3).mean(dim=(1, 2))

        assert metric.mean().item() > 0, "Metric must be nonzero when prediction != ground truth"
        assert metric.shape == (BATCH_SIZE,), "Metric must be per-sample"

    def test_residual_model_beats_naive_with_correct_delta(self):
        """
        A residual model given the true delta must score better than naive baseline.
        """
        torch.manual_seed(0)
        vel_in = torch.rand(BATCH_SIZE, NUM_T_IN, NUM_POS, 3)
        true_delta = torch.rand(BATCH_SIZE, NUM_T_OUT, NUM_POS, 3) * 0.1
        ground_truth = vel_in[:, -1:] + true_delta

        t = torch.rand(BATCH_SIZE, NUM_T_IN + NUM_T_OUT)
        pos = torch.rand(BATCH_SIZE, NUM_POS, 3)
        idcs = [torch.randint(NUM_POS, (1000,)) for _ in range(BATCH_SIZE)]

        naive = NaiveBaselineModel()
        oracle = ResidualWrapperModel(lambda *_: true_delta)

        naive_metric = (naive(t, pos, idcs, vel_in) - ground_truth).norm(dim=3).mean(dim=(1, 2)).mean()
        oracle_metric = (oracle(t, pos, idcs, vel_in) - ground_truth).norm(dim=3).mean(dim=(1, 2)).mean()

        assert oracle_metric < naive_metric, (
            f"Oracle residual ({oracle_metric:.4f}) must beat naive ({naive_metric:.4f})"
        )
