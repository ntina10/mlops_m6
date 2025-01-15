from m6_project.data import corrupt_mnist
import os
import torch
import pytest
from torch.utils.data import TensorDataset
from unittest.mock import patch

@pytest.mark.skipif(
    not os.path.exists("data/processed"),
    reason="Processed data not found."
)
@pytest.mark.skipif(
    not os.path.exists("data/raw"),
    reason="Raw data not found."
)
def test_data():
    train, test = corrupt_mnist()
    assert len(train) == 30000
    assert len(test) == 5000
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28)
            assert y in range(10)
    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(0,10)).all()
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0,10)).all()

def test_corrupt_mnist():
    # Mock data
    train_images = torch.rand(600, 1, 28, 28)
    train_targets = torch.randint(0, 10, (600,))
    test_images = torch.rand(100, 1, 28, 28)
    test_targets = torch.randint(0, 10, (100,))

    # Mock file loading
    with patch("torch.load", side_effect=[train_images, train_targets, test_images, test_targets]):
        train_set, test_set = corrupt_mnist()

        # Verify datasets
        assert isinstance(train_set, TensorDataset)
        assert isinstance(test_set, TensorDataset)
        assert len(train_set) == 600
        assert len(test_set) == 100

