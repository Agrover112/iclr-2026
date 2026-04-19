"""
Regression tests for the no-slip-mask wiring across CLI → model.

Two bugs from 2026-04-14 that these tests would have caught:

    1. `argparse.BooleanOptionalAction` silently treats any flag that starts
       with `--no-` as the NEGATION form of a shorter flag. The original
       `--no-slip-mask` thus parsed to False for BOTH the "on" and "off"
       invocations (`--no-slip-mask` → False, `--no-no-slip-mask` → False),
       yielding identical A/B metrics.

    2. Modal's `local_entrypoint` auto-generates `--no-<arg>` negations with
       the same hazard: a parameter literally named `no_slip_mask` collided
       with the auto-negation of a nonexistent `slip_mask`.

The fix is to always use an affirmative flag name (`--enforce-no-slip`).
These tests lock that in at three layers:

    (A) src/train.py argparse parses the flag to the right bool
    (B) FixedEGNNModel propagates the kwarg to the instance attribute
    (C) forward() applies the mask: predictions at idcs_airfoil are exactly
        zero when enforced, and toggling the flag changes the output.

Run with:
    /home/agrov/gram/bin/python -m pytest tests/test_no_slip_mask.py -v

(B) and (C) require `torch_scatter` loadable. On machines where the local
torch_scatter binary is ABI-incompatible (common on this repo's dev setup),
those tests auto-skip; they run green in the Modal container.
"""

from __future__ import annotations

import pytest
import torch

# --------------------------------------------------------------------------
# (A) argparse layer — the flag that actually failed in prod
# --------------------------------------------------------------------------

class TestArgparseNoSlipFlag:
    """Exercise the src/train.py argparse in isolation.

    These tests reconstruct the BooleanOptionalAction registration that
    src/train.py uses, so the regression is caught even if the training
    script is temporarily broken by other changes. A second test then
    verifies src/train.py registers the same flag correctly.
    """

    @staticmethod
    def _build_parser():
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument(
            '--enforce-no-slip', dest='no_slip_mask',
            action=argparse.BooleanOptionalAction, default=None,
        )
        return p

    def test_no_flag_leaves_value_as_none(self):
        """No flag → None, so the model falls through to its class default."""
        args = self._build_parser().parse_args([])
        assert args.no_slip_mask is None

    def test_enforce_no_slip_resolves_to_true(self):
        """--enforce-no-slip must set True. This is the bug from 2026-04-14:
        the old `--no-slip-mask` flag set False here."""
        args = self._build_parser().parse_args(['--enforce-no-slip'])
        assert args.no_slip_mask is True

    def test_no_enforce_no_slip_resolves_to_false(self):
        """--no-enforce-no-slip must set False."""
        args = self._build_parser().parse_args(['--no-enforce-no-slip'])
        assert args.no_slip_mask is False

    def test_old_broken_flag_name_demonstrates_argparse_hazard(self):
        """Locks in the CAUSE of the bug: argparse.BooleanOptionalAction on
        a flag whose name already starts with `--no-` silently parses both
        the bare flag and its auto-negation to False.

        If Python ever changes this behavior, this test will break and we
        can simplify the flag naming. Until then, keep using affirmative
        names (--enforce-no-slip) and never touch --no-... as a dest prefix.
        """
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument(
            '--no-slip-mask', action=argparse.BooleanOptionalAction, default=None,
        )
        assert p.parse_args(['--no-slip-mask']).no_slip_mask is False
        assert p.parse_args(['--no-no-slip-mask']).no_slip_mask is False

    def test_src_train_registers_enforce_no_slip(self):
        """End-to-end check against the real src/train.py parser — verifies
        the fix is actually wired in the live training script."""
        import subprocess, sys
        r = subprocess.run(
            [sys.executable, '-m', 'src.train', '--help'],
            capture_output=True, text=True,
        )
        assert '--enforce-no-slip' in r.stdout, (
            'src/train.py must expose --enforce-no-slip (affirmative form). '
            'Do not revert to --no-slip-mask — argparse will silently parse it to False.'
        )
        assert '--no-enforce-no-slip' in r.stdout, (
            'BooleanOptionalAction must also expose --no-enforce-no-slip for disabling.'
        )


# --------------------------------------------------------------------------
# Model-level tests (B + C) — require torch_scatter
# --------------------------------------------------------------------------

# Local dev env has an ABI-broken torch_scatter that raises OSError at import
# time (not ImportError), so importorskip isn't enough. Catch both.
try:
    from models.fixed_egnn.model import FixedEGNNModel
    _MODEL_IMPORTABLE = True
    _MODEL_SKIP_REASON = ""
except (ImportError, OSError) as e:
    _MODEL_IMPORTABLE = False
    _MODEL_SKIP_REASON = f"FixedEGNNModel unavailable locally: {type(e).__name__}: {e}"


@pytest.mark.skipif(not _MODEL_IMPORTABLE, reason=_MODEL_SKIP_REASON)
class TestModelFlagPropagation:
    """Layer B: the kwarg must land on the instance attribute."""

    def test_default_is_true(self):
        """FixedEGNNModel() with no kwarg → mask ON (physically correct)."""
        m = FixedEGNNModel()
        assert m.no_slip_mask is True

    def test_explicit_true_stored_on_instance(self):
        m = FixedEGNNModel(no_slip_mask=True)
        assert m.no_slip_mask is True

    def test_explicit_false_stored_on_instance(self):
        """The actual A/B failure mode: if this is False when we asked True,
        the mask never fires even though the CLI looked right."""
        m = FixedEGNNModel(no_slip_mask=False)
        assert m.no_slip_mask is False

    def test_none_kwarg_preserves_class_default(self):
        """Passing None (as src/train.py does when CLI omits the flag) must
        leave the class-level default intact."""
        m = FixedEGNNModel(no_slip_mask=None)
        assert m.no_slip_mask is True


@pytest.mark.skipif(not _MODEL_IMPORTABLE, reason=_MODEL_SKIP_REASON)
class TestForwardPassMaskBehavior:
    """Layer C: the mask must actually zero predictions at idcs_airfoil."""

    @staticmethod
    def _tiny_inputs(batch=1, n=200, k=4, n_airfoil=30, seed=0):
        """Small synthetic inputs, precomputed point_features + knn_graph
        so we don't need scipy/KDTree.

        Boundary points get nonzero input velocity on purpose — without the
        mask the network's zero-init decoder yields `next_vel = window[-1]`,
        which at the boundary is whatever we put there. That's how we can
        tell "masked" and "unmasked" forwards apart at initialization.
        """
        torch.manual_seed(seed)
        t   = torch.rand(batch, 10)
        pos = torch.rand(batch, n, 3)
        airfoil = torch.randperm(n)[:n_airfoil]
        idcs_airfoil = [airfoil.clone() for _ in range(batch)]
        # nonzero velocity at boundary so unmasked output is visibly nonzero
        velocity_in = torch.rand(batch, 5, n, 3) + 0.5
        # point_features: (B, N, 4) = udf_truncated(1) + udf_gradient(3)
        point_features = torch.rand(batch, n, 4)
        # knn_graph: (B, N, k) of valid indices in [0, N)
        knn_graph = torch.randint(0, n, (batch, n, k), dtype=torch.long)
        return t, pos, idcs_airfoil, velocity_in, point_features, knn_graph

    def test_masked_output_is_exactly_zero_at_airfoil(self):
        """With mask enforced, predictions at idcs_airfoil must be 0
        at every rollout step (output shape is (B, 5, N, 3))."""
        model = FixedEGNNModel(no_slip_mask=True).eval()
        t, pos, idcs, vel_in, pf, knn = self._tiny_inputs()

        with torch.no_grad():
            out = model(t, pos, idcs, vel_in, point_features=pf, knn_graph=knn)

        for b, idc in enumerate(idcs):
            boundary_preds = out[b, :, idc, :]
            assert torch.all(boundary_preds == 0.0), (
                f"Masked output had nonzero predictions at airfoil indices "
                f"(batch {b}, max abs={boundary_preds.abs().max().item()}). "
                "no_slip_mask=True must force exact zeros."
            )

    def test_unmasked_output_nonzero_at_airfoil(self):
        """With mask disabled and nonzero boundary input velocity, the
        initial-weights forward pass is (approximately) `window[-1]` at the
        boundary — which is nonzero by construction. Guards against a
        regression where somebody leaves the mask hardcoded."""
        model = FixedEGNNModel(no_slip_mask=False).eval()
        t, pos, idcs, vel_in, pf, knn = self._tiny_inputs()

        with torch.no_grad():
            out = model(t, pos, idcs, vel_in, point_features=pf, knn_graph=knn)

        for b, idc in enumerate(idcs):
            boundary_preds = out[b, :, idc, :]
            assert boundary_preds.abs().max().item() > 0.0, (
                "Unmasked output was exactly zero at airfoil — either the "
                "mask is being applied despite no_slip_mask=False, or the "
                "test inputs lost their nonzero boundary velocity."
            )

    def test_toggling_mask_changes_output(self):
        """A/B sanity: same seed, same weights, toggling the flag must
        produce different outputs. This is the test that would have caught
        the 2026-04-14 A/B where both runs returned identical metrics."""
        t, pos, idcs, vel_in, pf, knn = self._tiny_inputs(seed=1)

        def _forward(mask_flag: bool):
            torch.manual_seed(0)
            m = FixedEGNNModel(no_slip_mask=mask_flag).eval()
            with torch.no_grad():
                return m(t, pos, idcs, vel_in, point_features=pf, knn_graph=knn)

        out_on = _forward(True)
        out_off = _forward(False)

        assert not torch.allclose(out_on, out_off), (
            "Masked and unmasked forwards produced identical output — "
            "the flag is not reaching the model.forward() code path. "
            "(This is the exact signature of the 2026-04-14 A/B bug.)"
        )
