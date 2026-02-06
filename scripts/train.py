"""
scripts/train.py

Training entry point.

uv run python scripts/train.py --help

uv run python scripts/train.py \
    policy:libero \
    --dataset.repo-id lerobot/droid_100 \
    --policy.repo-id brandonyang/vvla-libero-pi05
"""

from dataclasses import dataclass

import tyro

from vvla.datasets import (
    LeRobotDatasetConfig,
    make_dataloader,
    make_dataset,
)
from vvla.policies import PolicyConfig, make_policy


@dataclass
class TrainConfig:
    dataset: LeRobotDatasetConfig
    policy: PolicyConfig


def main(config: TrainConfig) -> None:
    dataset = make_dataset(config.dataset)
    policy = make_policy(config.policy, norm_stats=dataset.norm_stats, mode="train")
    dataloader = make_dataloader(  # noqa: F841
        config.dataset, dataset, transforms=policy.input_transforms
    )

    print("Training setup complete.")

    # Save checkpoint
    policy.save_pretrained("output/checkpoint")
    # policy.push_to_hub()


if __name__ == "__main__":
    main(tyro.cli(TrainConfig, config=(tyro.conf.CascadeSubcommandArgs,)))
