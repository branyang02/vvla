from pathlib import Path

import torch
import tyro
from typing_extensions import Annotated

from vvla.models import make_model
from vvla.policies.base_policy import BasePolicy
from vvla.policies.dummy_policy import DummyPolicy, DummyPolicyConfig
from vvla.policies.libero_policy import LiberoPolicy, LiberoPolicyConfig
from vvla.policies.metaworld_policy import MetaworldPolicy, MetaworldPolicyConfig
from vvla.transforms import make_transforms
from vvla.transforms.transforms import NormStats

PolicyConfig = (
    Annotated[LiberoPolicyConfig, tyro.conf.subcommand("libero")]
    | Annotated[MetaworldPolicyConfig, tyro.conf.subcommand("metaworld")]
    | Annotated[DummyPolicyConfig, tyro.conf.subcommand("dummy")]
)


def make_policy(config: PolicyConfig, norm_stats: dict[str, NormStats]) -> BasePolicy:
    """Create a policy from config."""

    if isinstance(config, DummyPolicyConfig):
        return DummyPolicy(config=config)

    model = make_model(config.model)
    input_transforms, output_transforms = make_transforms(config.transforms, norm_stats)

    if isinstance(config, LiberoPolicyConfig):
        return LiberoPolicy(
            config=config,
            model=model,
            input_transforms=input_transforms,
            output_transforms=output_transforms,
        )
    elif isinstance(config, MetaworldPolicyConfig):
        return MetaworldPolicy(
            config=config,
            model=model,
            input_transforms=input_transforms,
            output_transforms=output_transforms,
        )
    raise ValueError(f"Unknown policy config: {config}")


def load_policy(path: str | Path) -> BasePolicy:
    """Load a policy from checkpoint file."""
    checkpoint = torch.load(path, weights_only=False)
    policy_type = checkpoint["policy_type"]

    if policy_type == "libero":
        return LiberoPolicy.load(path)
    elif policy_type == "metaworld":
        return MetaworldPolicy.load(path)
    raise ValueError(f"Unknown policy type: {policy_type}")


__all__ = [
    "PolicyConfig",
    "make_policy",
    "load_policy",
]
