"""
scripts/train.py

Training entry point.

uv run python scripts/train.py --help
"""

from dataclasses import dataclass
from pathlib import Path

import tyro

from vlla.datasets import (
    LeRobotDatasetConfig,
    make_dataloader,
    make_dataset,
)
from vlla.policies import PolicyConfig, make_policy


@dataclass
class TrainConfig:
    dataset: LeRobotDatasetConfig
    policy: PolicyConfig

    output_dir: Path = Path("output")
    checkpoint_name: str = "policy.pt"


def main(config: TrainConfig) -> None:
    dataset = make_dataset(config.dataset)
    policy = make_policy(config.policy, norm_stats=dataset.norm_stats)
    dataloader = make_dataloader(  # noqa: F841
        config.dataset, dataset, transforms=policy.input_transforms
    )

    print("Training setup complete.")

    # Save checkpoint
    config.output_dir.mkdir(parents=True, exist_ok=True)
    policy.save(config.output_dir / config.checkpoint_name)
    print(f"Saved checkpoint to {config.output_dir / config.checkpoint_name}")


if __name__ == "__main__":
    main(tyro.cli(TrainConfig, config=(tyro.conf.CascadeSubcommandArgs,)))
