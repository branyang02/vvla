"""
uv run pytest tests/test_datasets.py
"""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from vlla.datasets import (
    LeRobotDatasetConfig,
    TransformedDataset,
    make_dataloader,
    make_dataset,
)
from vlla.datasets.data_loader import Dataset
from vlla.transforms.transforms import NormStats

#### Fixtures ####


class MockDataset(torch.utils.data.Dataset):
    """A simple mock dataset for testing."""

    def __init__(self, size: int = 100, feature_dim: int = 10):
        self.size = size
        self.feature_dim = feature_dim
        self.data = [
            {
                "observation": np.random.randn(feature_dim).astype(np.float32),
                "action": np.random.randn(4).astype(np.float32),
            }
            for _ in range(size)
        ]

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self) -> int:
        return self.size


class MockLeRobotDataset(torch.utils.data.Dataset):
    """Mock LeRobotDataset with meta.stats for testing Dataset wrapper."""

    def __init__(self, size: int = 100):
        self.size = size
        self.data = [
            {
                "observation.state": np.random.randn(10).astype(np.float32),
                "action": np.random.randn(4).astype(np.float32),
            }
            for _ in range(size)
        ]
        # Mock the meta.stats structure that LeRobotDataset provides
        self.meta = MockMeta()

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self) -> int:
        return self.size


class MockMeta:
    """Mock meta object with stats."""

    def __init__(self):
        self.stats = {
            "action": {
                "mean": np.array([0.0, 0.1, 0.2, 0.3]),
                "std": np.array([1.0, 1.1, 1.2, 1.3]),
                "min": np.array([-2.0, -2.1, -2.2, -2.3]),
                "max": np.array([2.0, 2.1, 2.2, 2.3]),
            },
            "observation.state": {
                "mean": np.zeros(10),
                "std": np.ones(10),
                "min": np.full(10, -3.0),
                "max": np.full(10, 3.0),
            },
        }


@pytest.fixture
def mock_dataset() -> MockDataset:
    return MockDataset(size=50, feature_dim=10)


@pytest.fixture
def mock_lerobot_dataset() -> MockLeRobotDataset:
    return MockLeRobotDataset(size=50)


# ==============================================================================
# TransformedDataset Tests
# ==============================================================================


def test_transformed_dataset_length(mock_dataset):
    transformed = TransformedDataset(mock_dataset)
    assert len(transformed) == len(mock_dataset)


def test_transformed_dataset_getitem_no_transform(mock_dataset):
    transformed = TransformedDataset(mock_dataset)
    item = transformed[0]
    assert "observation" in item
    assert "action" in item
    np.testing.assert_array_equal(item["observation"], mock_dataset[0]["observation"])


def test_transformed_dataset_applies_transform(mock_dataset):
    """Test that transforms are applied to items."""

    class AddOneTransform:
        def __call__(self, data: dict) -> dict:
            return {k: v + 1 for k, v in data.items()}

    transformed = TransformedDataset(mock_dataset, transforms=[AddOneTransform()])
    item = transformed[0]
    original = mock_dataset[0]

    np.testing.assert_array_almost_equal(
        item["observation"], original["observation"] + 1
    )
    np.testing.assert_array_almost_equal(item["action"], original["action"] + 1)


def test_transformed_dataset_composes_transforms(mock_dataset):
    """Test that multiple transforms are composed in order."""

    class MultiplyTransform:
        def __init__(self, factor: float):
            self.factor = factor

        def __call__(self, data: dict) -> dict:
            return {k: v * self.factor for k, v in data.items()}

    transforms = [MultiplyTransform(2.0), MultiplyTransform(3.0)]
    transformed = TransformedDataset(mock_dataset, transforms=transforms)
    item = transformed[0]
    original = mock_dataset[0]

    # Should multiply by 2 then by 3 = multiply by 6
    np.testing.assert_array_almost_equal(
        item["observation"], original["observation"] * 6.0
    )


# ==============================================================================
# Dataset Wrapper Tests (with mock LeRobotDataset)
# ==============================================================================


def test_dataset_length(mock_lerobot_dataset):
    # Patch isinstance check for LeRobotDataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Use monkeypatch to make isinstance work
    MockLeRobotDataset.__bases__ = (LeRobotDataset,)
    try:
        dataset = Dataset(mock_lerobot_dataset)
        assert len(dataset) == len(mock_lerobot_dataset)
    finally:
        MockLeRobotDataset.__bases__ = (torch.utils.data.Dataset,)


def test_dataset_getitem(mock_lerobot_dataset):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    MockLeRobotDataset.__bases__ = (LeRobotDataset,)
    try:
        dataset = Dataset(mock_lerobot_dataset)
        item = dataset[0]
        assert "observation.state" in item
        assert "action" in item
    finally:
        MockLeRobotDataset.__bases__ = (torch.utils.data.Dataset,)


def test_dataset_norm_stats_computed(mock_lerobot_dataset):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    MockLeRobotDataset.__bases__ = (LeRobotDataset,)
    try:
        dataset = Dataset(mock_lerobot_dataset)
        assert hasattr(dataset, "norm_stats")
        assert isinstance(dataset.norm_stats, dict)
        assert "action" in dataset.norm_stats
        assert "observation.state" in dataset.norm_stats
    finally:
        MockLeRobotDataset.__bases__ = (torch.utils.data.Dataset,)


def test_dataset_norm_stats_structure(mock_lerobot_dataset):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    MockLeRobotDataset.__bases__ = (LeRobotDataset,)
    try:
        dataset = Dataset(mock_lerobot_dataset)
        action_stats = dataset.norm_stats["action"]

        assert isinstance(action_stats, NormStats)
        assert action_stats.mean is not None
        assert action_stats.std is not None
        assert action_stats.q01 is not None  # mapped from min
        assert action_stats.q99 is not None  # mapped from max

        np.testing.assert_array_almost_equal(
            action_stats.mean, np.array([0.0, 0.1, 0.2, 0.3])
        )
        np.testing.assert_array_almost_equal(
            action_stats.std, np.array([1.0, 1.1, 1.2, 1.3])
        )
    finally:
        MockLeRobotDataset.__bases__ = (torch.utils.data.Dataset,)


def test_dataset_unsupported_type_raises(mock_dataset):
    """Test that unsupported dataset types raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match="not implemented for dataset type"):
        Dataset(mock_dataset)


# ==============================================================================
# LeRobotDatasetConfig Tests
# ==============================================================================


def test_lerobot_dataset_config_defaults():
    config = LeRobotDatasetConfig(repo_id="test/repo")
    assert config.repo_id == "test/repo"
    assert config.batch_size == 64
    assert config.shuffle is True


def test_lerobot_dataset_config_custom():
    config = LeRobotDatasetConfig(repo_id="custom/repo", batch_size=32, shuffle=False)
    assert config.repo_id == "custom/repo"
    assert config.batch_size == 32
    assert config.shuffle is False


# ==============================================================================
# make_dataset Tests
# ==============================================================================


def _create_mock_lerobot():
    """Create a mock that looks like LeRobotDataset."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    mock = MockLeRobotDataset(size=50)
    MockLeRobotDataset.__bases__ = (LeRobotDataset,)
    return mock


def test_make_dataset_returns_dataset():
    """Test that make_dataset returns a Dataset wrapper."""
    mock = _create_mock_lerobot()
    try:
        with patch("vlla.datasets.LeRobotDataset", return_value=mock):
            config = LeRobotDatasetConfig(repo_id="test/repo")
            dataset = make_dataset(config)
            assert isinstance(dataset, Dataset)
    finally:
        MockLeRobotDataset.__bases__ = (torch.utils.data.Dataset,)


def test_make_dataset_has_norm_stats():
    """Test that make_dataset returns a Dataset with norm_stats."""
    mock = _create_mock_lerobot()
    try:
        with patch("vlla.datasets.LeRobotDataset", return_value=mock):
            config = LeRobotDatasetConfig(repo_id="test/repo")
            dataset = make_dataset(config)
            assert hasattr(dataset, "norm_stats")
            assert "action" in dataset.norm_stats
            assert "observation.state" in dataset.norm_stats
    finally:
        MockLeRobotDataset.__bases__ = (torch.utils.data.Dataset,)


def test_make_dataset_preserves_length():
    """Test that make_dataset preserves the underlying dataset length."""
    mock = _create_mock_lerobot()
    try:
        with patch("vlla.datasets.LeRobotDataset", return_value=mock):
            config = LeRobotDatasetConfig(repo_id="test/repo")
            dataset = make_dataset(config)
            assert len(dataset) == 50  # MockLeRobotDataset default size
    finally:
        MockLeRobotDataset.__bases__ = (torch.utils.data.Dataset,)


def test_make_dataset_items_accessible():
    """Test that items from make_dataset are accessible."""
    mock = _create_mock_lerobot()
    try:
        with patch("vlla.datasets.LeRobotDataset", return_value=mock):
            config = LeRobotDatasetConfig(repo_id="test/repo")
            dataset = make_dataset(config)
            item = dataset[0]
            assert "observation.state" in item
            assert "action" in item
    finally:
        MockLeRobotDataset.__bases__ = (torch.utils.data.Dataset,)


def test_make_dataset_passes_repo_id():
    """Test that make_dataset passes repo_id to LeRobotDataset."""
    mock = _create_mock_lerobot()
    try:
        with patch("vlla.datasets.LeRobotDataset", return_value=mock) as mock_cls:
            config = LeRobotDatasetConfig(repo_id="my/custom-repo")
            make_dataset(config)
            mock_cls.assert_called_once_with(repo_id="my/custom-repo")
    finally:
        MockLeRobotDataset.__bases__ = (torch.utils.data.Dataset,)


# ==============================================================================
# DataLoader Tests
# ==============================================================================


def test_make_dataloader_returns_dataloader(mock_lerobot_dataset):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    MockLeRobotDataset.__bases__ = (LeRobotDataset,)
    try:
        dataset = Dataset(mock_lerobot_dataset)
        config = LeRobotDatasetConfig(repo_id="test/repo", batch_size=8, shuffle=False)
        dataloader = make_dataloader(config, dataset)

        assert isinstance(dataloader, torch.utils.data.DataLoader)
        assert dataloader.batch_size == 8
    finally:
        MockLeRobotDataset.__bases__ = (torch.utils.data.Dataset,)


def test_make_dataloader_with_transforms(mock_lerobot_dataset):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    MockLeRobotDataset.__bases__ = (LeRobotDataset,)
    try:
        dataset = Dataset(mock_lerobot_dataset)
        config = LeRobotDatasetConfig(repo_id="test/repo", batch_size=4, shuffle=False)

        call_count = [0]

        class CountingTransform:
            def __call__(self, data: dict) -> dict:
                call_count[0] += 1
                return data

        dataloader = make_dataloader(config, dataset, transforms=[CountingTransform()])

        # Get one batch to trigger transforms
        batch = next(iter(dataloader))  # noqa: F841
        assert call_count[0] == 4  # batch_size items transformed
    finally:
        MockLeRobotDataset.__bases__ = (torch.utils.data.Dataset,)


def test_make_dataloader_iterates(mock_lerobot_dataset):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    MockLeRobotDataset.__bases__ = (LeRobotDataset,)
    try:
        dataset = Dataset(mock_lerobot_dataset)
        config = LeRobotDatasetConfig(repo_id="test/repo", batch_size=10, shuffle=False)
        dataloader = make_dataloader(config, dataset)

        batches = list(dataloader)
        assert len(batches) == 5  # 50 items / 10 batch_size
    finally:
        MockLeRobotDataset.__bases__ = (torch.utils.data.Dataset,)
