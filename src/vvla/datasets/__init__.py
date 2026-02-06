from collections.abc import Sequence
from dataclasses import dataclass

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from vvla.datasets.data_loader import Dataset, TransformedDataset
from vvla.transforms.transforms import DataTransformFn


@dataclass
class LeRobotDatasetConfig:
    repo_id: str
    batch_size: int = 64
    shuffle: bool = True


def make_dataset(config: LeRobotDatasetConfig) -> Dataset:
    """
    Create a dataset instance from the given configuration.

    Currently, only `LeRobotDataset` is supported.
    """
    return Dataset(LeRobotDataset(repo_id=config.repo_id))


def make_dataloader(
    config: LeRobotDatasetConfig,
    dataset: Dataset,
    transforms: Sequence[DataTransformFn] = (),
) -> torch.utils.data.DataLoader:
    dataset = TransformedDataset(dataset, transforms)
    return torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=config.shuffle
    )


__all__ = [
    "LeRobotDatasetConfig",
    "make_dataset",
    "make_dataloader",
]
