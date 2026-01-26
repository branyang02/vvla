from collections.abc import Sequence

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from vlla.transforms.transforms import DataTransformFn, NormStats, compose


class TransformedDataset(torch.utils.data.Dataset):
    """Wraps a dataset and applies transforms to each item."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        transforms: Sequence[DataTransformFn] = (),
    ):
        self._dataset = dataset
        self._transform = compose(transforms) if transforms else lambda x: x

    def __getitem__(self, index: int):
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class Dataset(torch.utils.data.Dataset):
    """Dataset wrapper that calculates normalization statistics on init."""

    def __init__(self, dataset: torch.utils.data.Dataset):
        self._dataset = dataset
        self.norm_stats = self._get_norm_stats()

    def _get_norm_stats(self) -> dict[str, NormStats]:
        if isinstance(self._dataset, LeRobotDataset):
            lerobot_meta_stats = self._dataset.meta.stats
            # default lerobot keys to normalize/unnormalize
            target_keys = ["action", "observation.state"]

            stats_map = {}
            for key in target_keys:
                if key not in lerobot_meta_stats:
                    print(f"Warning: Key '{key}' not found in dataset stats.")
                    continue

                raw = lerobot_meta_stats[key]

                # Helper to ensure we have a numpy array
                def to_np(x):
                    return np.array(x, dtype=np.float32)

                # Map LeRobot 'min'/'max' to 'q01'/'q99'
                stats_map[key] = NormStats(
                    mean=to_np(raw["mean"]),
                    std=to_np(raw["std"]),
                    q01=to_np(raw["min"]),
                    q99=to_np(raw["max"]),
                )

            return stats_map

        raise NotImplementedError(
            f"Normalization statistics calculation not implemented for dataset type {type(self._dataset)}."
        )

    def __getitem__(self, index: int):
        return self._dataset[index]

    def __len__(self) -> int:
        return len(self._dataset)
