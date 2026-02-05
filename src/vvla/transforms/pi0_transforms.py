from dataclasses import dataclass
from typing import Sequence

from vvla.transforms.transforms import (
    DataTransformFn,
    Normalize,
    NormStats,
    Unnormalize,
)


@dataclass
class Pi0TransformsConfig:
    """Transforms for Pi0 policy - normalize/unnormalize only."""

    def build(
        self, norm_stats: dict[str, NormStats]
    ) -> tuple[Sequence[DataTransformFn], Sequence[DataTransformFn]]:
        input_transforms = [
            Normalize(norm_stats=norm_stats),
        ]
        output_transforms = [
            Unnormalize(norm_stats=norm_stats),
        ]

        return input_transforms, output_transforms
