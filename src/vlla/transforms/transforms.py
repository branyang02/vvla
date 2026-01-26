from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

DataDict = dict[str, np.ndarray]


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """Applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for t in self.transforms:
            data = t(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclass
class NormStats:
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray | None
    q99: np.ndarray | None


@dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: dict[str, NormStats]

    def __call__(self, data: DataDict) -> DataDict:
        for key, stats in self.norm_stats.items():
            if key not in data:
                continue
            normalized = (data[key] - stats.mean) / (stats.std + 1e-8)
            data[key] = normalized
        return data


@dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: dict[str, NormStats]

    def __call__(self, data: DataDict) -> DataDict:
        for key, stats in self.norm_stats.items():
            if key not in data:
                continue
            data[key] = data[key] * stats.std + stats.mean
        return data


@dataclass(frozen=True)
class ResizeTransform(DataTransformFn):
    height: int = 224
    width: int = 224

    def __call__(self, data: DataDict) -> DataDict:
        return data
