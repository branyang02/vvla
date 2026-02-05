from typing import Annotated, Sequence

import tyro

from vvla.transforms.pi0_transforms import Pi0TransformsConfig
from vvla.transforms.pi05_transforms import Pi05TransformsConfig
from vvla.transforms.transforms import DataTransformFn, NormStats

TransformsConfig = (
    Annotated[Pi05TransformsConfig, tyro.conf.subcommand("pi05")]
    | Annotated[Pi0TransformsConfig, tyro.conf.subcommand("pi0")]
)


def make_transforms(
    config: TransformsConfig, norm_stats: dict[str, NormStats]
) -> tuple[Sequence[DataTransformFn], Sequence[DataTransformFn]]:
    """Create input and output transforms from config."""
    return config.build(norm_stats=norm_stats)


__all__ = [
    "TransformsConfig",
    "make_transforms",
]
