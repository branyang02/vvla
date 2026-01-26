from dataclasses import dataclass, field
from typing import Sequence

from vlla.transforms.transforms import (
    DataTransformFn,
    Normalize,
    NormStats,
    Unnormalize,
)


@dataclass
class Pi05TransformsConfig:
    """Transforms for Pi0.5 policy - normalize/unnormalize + image resize."""

    image_keys: list[str] = field(default_factory=lambda: ["observation.image"])
    image_height: int = 224
    image_width: int = 224

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
