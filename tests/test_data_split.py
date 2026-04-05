"""
Unit tests for src/data.py split logic.

Covers bugs found and fixed:
  1. Val != Test (odd remainder going to test instead of train)
  2. Wrong assertion (file count vs geometry count)
  3. Assertion failure with data_fraction < 1.0
  4. Geometry leakage across splits
  5. Empty val/test on very small datasets
  6. sim_key parsing: split by simulation, not base geometry_id
"""

import os
import tempfile
import pytest

from src.data import split_by_geometry, sim_key


def make_fake_data_dir(sim_keys: list[str], chunks_per_sim: int = 5) -> str:
    """
    Create a temp directory with empty .npz files.

    Each sim_key (e.g. "1021_10") gets `chunks_per_sim` files:
      {sim_key}-0.npz, {sim_key}-1.npz, ...

    Reflects real naming: {geometry_id}_{sim_id}-{chunk_id}.npz
    """
    tmp_dir = tempfile.mkdtemp()
    for key in sim_keys:
        for chunk_id in range(chunks_per_sim):
            fname = f"{key}-{chunk_id}.npz"
            open(os.path.join(tmp_dir, fname), "w").close()
    return tmp_dir


def make_multi_sim_data_dir(geom_ids: list[str], sims_per_geom: int, chunks_per_sim: int = 5) -> str:
    """
    Create fake data with multiple sim_ids per geometry_id, matching real dataset
    structure where e.g. geom 1021 has sims 1, 10, 11, 12, ...
    """
    tmp_dir = tempfile.mkdtemp()
    for geom_id in geom_ids:
        for sim_id in range(sims_per_geom):
            for chunk_id in range(chunks_per_sim):
                fname = f"{geom_id}_{sim_id}-{chunk_id}.npz"
                open(os.path.join(tmp_dir, fname), "w").close()
    return tmp_dir


# ---------------------------------------------------------------------------
# sim_key parsing
# ---------------------------------------------------------------------------

class TestSimKey:
    def test_extracts_geom_and_sim(self):
        assert sim_key("1021_10-3.npz") == "1021_10"

    def test_works_with_full_path(self):
        assert sim_key("/data/1021_10-3.npz") == "1021_10"

    def test_chunk_zero(self):
        assert sim_key("3006_1-0.npz") == "3006_1"

    def test_different_sims_same_geom(self):
        assert sim_key("1021_1-0.npz") != sim_key("1021_10-0.npz")

    def test_same_sim_different_chunks(self):
        assert sim_key("1021_10-0.npz") == sim_key("1021_10-4.npz")


# ---------------------------------------------------------------------------
# Bug 1: val == test (odd remainder must go to train, not test)
# ---------------------------------------------------------------------------

class TestValEqualsTest:
    def test_val_equals_test_odd_remainder(self):
        """When val_test_total is odd, test must not receive the extra simulation."""
        # 81 sims: int(0.7*81)=56, remainder=25 (odd) → val=test=12, train=57
        sim_keys = [f"g_{i}" for i in range(81)]
        tmp_dir = make_fake_data_dir(sim_keys, chunks_per_sim=1)
        splits = split_by_geometry(tmp_dir, train_ratio=0.7, val_ratio=0.15, seed=42)
        assert len(splits['val']) == len(splits['test']), (
            f"val={len(splits['val'])} != test={len(splits['test'])} (odd remainder went to test)"
        )

    def test_val_equals_test_even_remainder(self):
        """When val_test_total is even, val and test should still be equal."""
        sim_keys = [f"g_{i}" for i in range(80)]
        tmp_dir = make_fake_data_dir(sim_keys, chunks_per_sim=1)
        splits = split_by_geometry(tmp_dir, train_ratio=0.7, val_ratio=0.15, seed=42)
        assert len(splits['val']) == len(splits['test'])

    def test_val_equals_test_multiple_chunks_per_sim(self):
        """Val and test simulation counts must be equal even with multiple chunks per sim."""
        sim_keys = [f"g_{i}" for i in range(45)]
        tmp_dir = make_fake_data_dir(sim_keys, chunks_per_sim=5)
        splits = split_by_geometry(tmp_dir, train_ratio=0.7, val_ratio=0.15, seed=42)
        val_sims = set(sim_key(f) for f in splits['val'])
        test_sims = set(sim_key(f) for f in splits['test'])
        assert len(val_sims) == len(test_sims)


# ---------------------------------------------------------------------------
# Bug 2: assertion uses file count, not simulation count
# ---------------------------------------------------------------------------

class TestFileCountAssertion:
    def test_assertion_does_not_compare_files_to_sim_count(self):
        """
        The old assertion compared total files against number of sim keys (N),
        which always fails when chunks_per_sim > 1.
        """
        sim_keys = [f"g_{i}" for i in range(50)]
        tmp_dir = make_fake_data_dir(sim_keys, chunks_per_sim=5)  # 250 files, 50 sims
        splits = split_by_geometry(tmp_dir, train_ratio=0.7, val_ratio=0.15, seed=42)
        total = len(splits['train']) + len(splits['val']) + len(splits['test'])
        assert total == 250

    def test_assertion_single_chunk_per_sim(self):
        """Sanity check: still works when chunks_per_sim == 1."""
        sim_keys = [f"g_{i}" for i in range(30)]
        tmp_dir = make_fake_data_dir(sim_keys, chunks_per_sim=1)
        splits = split_by_geometry(tmp_dir, train_ratio=0.7, val_ratio=0.15, seed=42)
        total = len(splits['train']) + len(splits['val']) + len(splits['test'])
        assert total == 30


# ---------------------------------------------------------------------------
# Bug 3: assertion must work with data_fraction < 1.0
# ---------------------------------------------------------------------------

class TestDataFraction:
    def test_assertion_passes_with_data_fraction(self):
        """Old assertion compared against ALL sims even when only a fraction were used."""
        sim_keys = [f"g_{i}" for i in range(100)]
        tmp_dir = make_fake_data_dir(sim_keys, chunks_per_sim=3)
        splits = split_by_geometry(tmp_dir, train_ratio=0.7, val_ratio=0.15, seed=42, data_fraction=0.5)
        total = len(splits['train']) + len(splits['val']) + len(splits['test'])
        assert total == 50 * 3

    def test_val_equals_test_with_data_fraction(self):
        sim_keys = [f"g_{i}" for i in range(100)]
        tmp_dir = make_fake_data_dir(sim_keys, chunks_per_sim=1)
        splits = split_by_geometry(tmp_dir, train_ratio=0.7, val_ratio=0.15, seed=42, data_fraction=0.5)
        assert len(splits['val']) == len(splits['test'])

    def test_splits_sum_to_selected_not_total(self):
        sim_keys = [f"g_{i}" for i in range(200)]
        tmp_dir = make_fake_data_dir(sim_keys, chunks_per_sim=2)
        splits = split_by_geometry(tmp_dir, train_ratio=0.7, val_ratio=0.15, seed=42, data_fraction=0.1)
        total = len(splits['train']) + len(splits['val']) + len(splits['test'])
        assert total < 200 * 2
        assert total > 0


# ---------------------------------------------------------------------------
# Bug 4: no simulation leakage across splits
# ---------------------------------------------------------------------------

class TestNoSimulationLeakage:
    def test_simulation_does_not_appear_in_multiple_splits(self):
        """No simulation key may appear in more than one split."""
        sim_keys = [f"geom{i // 8}_{i % 8}" for i in range(160)]  # 20 geoms × 8 sims
        tmp_dir = make_fake_data_dir(sim_keys, chunks_per_sim=5)
        splits = split_by_geometry(tmp_dir, train_ratio=0.7, val_ratio=0.15, seed=42)

        train_sims = set(sim_key(f) for f in splits['train'])
        val_sims = set(sim_key(f) for f in splits['val'])
        test_sims = set(sim_key(f) for f in splits['test'])

        assert train_sims.isdisjoint(val_sims), "Simulation leakage: train ∩ val"
        assert train_sims.isdisjoint(test_sims), "Simulation leakage: train ∩ test"
        assert val_sims.isdisjoint(test_sims), "Simulation leakage: val ∩ test"

    def test_chunks_of_same_sim_stay_together(self):
        """All 5 chunks of a simulation must land in the same split."""
        # 22 geom_ids × 8 sims each = 176 simulations, 5 chunks each
        tmp_dir = make_multi_sim_data_dir(
            geom_ids=[str(1000 + i) for i in range(22)],
            sims_per_geom=8,
            chunks_per_sim=5,
        )
        splits = split_by_geometry(tmp_dir, train_ratio=0.7, val_ratio=0.15, seed=42)

        # For every sim_key, all its chunks must be in exactly one split
        from collections import defaultdict
        sim_to_splits = defaultdict(set)
        for split_name, file_list in splits.items():
            for f in file_list:
                sim_to_splits[sim_key(f)].add(split_name)

        leaking = {k: v for k, v in sim_to_splits.items() if len(v) > 1}
        assert not leaking, f"Chunks split across splits for: {leaking}"

    def test_different_sims_of_same_geom_can_be_in_different_splits(self):
        """
        Different sim_ids of the same geometry_id are independent geometries
        and are allowed (expected) to appear in different splits.
        """
        # 4 geom_ids × 40 sims each = 160 simulations
        tmp_dir = make_multi_sim_data_dir(
            geom_ids=["1021", "1022", "1023", "1024"],
            sims_per_geom=40,
            chunks_per_sim=5,
        )
        splits = split_by_geometry(tmp_dir, train_ratio=0.7, val_ratio=0.15, seed=42)

        # All 4 geom_ids should appear in train (they have enough sims to spread)
        train_base_geoms = set(sim_key(f).split("_")[0] for f in splits['train'])
        val_base_geoms = set(sim_key(f).split("_")[0] for f in splits['val'])
        # At least one base geom_id should appear in both train and val
        assert train_base_geoms & val_base_geoms, (
            "Expected same geom_id to appear in multiple splits via different sim_ids"
        )


# ---------------------------------------------------------------------------
# Bug 5: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_very_small_dataset_does_not_crash(self):
        """With n=3 sims and train_ratio=0.7, val/test may be empty but must not crash."""
        sim_keys = ["g_0", "g_1", "g_2"]
        tmp_dir = make_fake_data_dir(sim_keys, chunks_per_sim=1)
        splits = split_by_geometry(tmp_dir, train_ratio=0.7, val_ratio=0.15, seed=42)
        total = len(splits['train']) + len(splits['val']) + len(splits['test'])
        assert total <= 3

    def test_reproducibility(self):
        """Same seed always produces the same split."""
        sim_keys = [f"g_{i}" for i in range(100)]
        tmp_dir = make_fake_data_dir(sim_keys, chunks_per_sim=1)
        splits_a = split_by_geometry(tmp_dir, train_ratio=0.7, val_ratio=0.15, seed=42)
        splits_b = split_by_geometry(tmp_dir, train_ratio=0.7, val_ratio=0.15, seed=42)
        assert sorted(splits_a['train']) == sorted(splits_b['train'])
        assert sorted(splits_a['val']) == sorted(splits_b['val'])
        assert sorted(splits_a['test']) == sorted(splits_b['test'])

    def test_different_seeds_produce_different_splits(self):
        """Different seeds should produce different splits."""
        sim_keys = [f"g_{i}" for i in range(100)]
        tmp_dir = make_fake_data_dir(sim_keys, chunks_per_sim=1)
        splits_42 = split_by_geometry(tmp_dir, train_ratio=0.7, val_ratio=0.15, seed=42)
        splits_99 = split_by_geometry(tmp_dir, train_ratio=0.7, val_ratio=0.15, seed=99)
        assert sorted(splits_42['train']) != sorted(splits_99['train'])
