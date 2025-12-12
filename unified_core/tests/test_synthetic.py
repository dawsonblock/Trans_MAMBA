"""Tests for synthetic tasks."""

import os
import sys

import torch

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)


def test_copy_memory_dataset():
    """Test CopyMemory dataset."""
    from synthetic import CopyMemoryDataset

    ds = CopyMemoryDataset(
        seq_len=64,
        delay=16,
        vocab_size=16,
        num_samples=100,
    )

    x, y, mask = ds[0]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert x.shape == (64,)
    assert y.shape == (64,)
    assert mask.shape == (64,)
    assert mask.sum() > 0
    print("✓ CopyMemoryDataset")


def test_assoc_recall_dataset():
    """Test AssocRecall dataset."""
    from synthetic import AssocRecallDataset

    ds = AssocRecallDataset(
        seq_len=64,
        num_pairs=4,
        vocab_size=16,
        num_samples=100,
    )

    x, y, mask = ds[0]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert x.shape == (64,)
    assert y.shape == (64,)
    assert mask.sum() == 1.0
    print("✓ AssocRecallDataset")


def test_selective_copy_dataset():
    """Test SelectiveCopy dataset."""
    from synthetic import SelectiveCopyDataset

    ds = SelectiveCopyDataset(
        seq_len=64,
        num_markers=3,
        vocab_size=16,
        num_samples=100,
    )

    x, y, mask = ds[0]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert x.shape == (64,)
    assert y.shape == (64,)
    assert mask.sum() > 0
    print("✓ SelectiveCopyDataset")


def test_induction_head_dataset():
    """Test InductionHead dataset."""
    from synthetic import InductionHeadDataset

    ds = InductionHeadDataset(
        seq_len=64,
        pattern_len=2,
        vocab_size=16,
        num_samples=100,
    )

    x, y, mask = ds[0]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert x.shape == (64,)
    assert y.shape == (64,)
    assert mask.sum() == 1.0
    print("✓ InductionHeadDataset")


def test_get_dataset_factory():
    """Test dataset factory function."""
    from synthetic import SyntheticTaskConfig, get_dataset

    cfg = SyntheticTaskConfig(
        seq_len=64, vocab_size=16, num_samples=100, delay=16
    )

    ds = get_dataset("copy_memory", cfg)
    assert len(ds) == 100

    ds = get_dataset("assoc_recall", cfg)
    assert len(ds) == 100

    print("✓ get_dataset factory")


def run_all_tests():
    """Run all synthetic tests."""
    print("\n" + "=" * 50)
    print("Running Synthetic Task Tests")
    print("=" * 50 + "\n")

    test_copy_memory_dataset()
    test_assoc_recall_dataset()
    test_selective_copy_dataset()
    test_induction_head_dataset()
    test_get_dataset_factory()

    print("\n" + "=" * 50)
    print("All synthetic tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()
