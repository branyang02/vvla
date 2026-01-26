"""
scripts/eval.py

uv run python scripts/eval.py --help
"""

from dataclasses import dataclass, field
from pathlib import Path

import tyro
from typing_extensions import Annotated

from vlla.datasets import LeRobotDatasetConfig, make_dataset
from vlla.policies import PolicyConfig, load_policy, make_policy
from vlla.serving.websocket_policy_server import WebsocketPolicyServer


@dataclass
class EvalFromCheckpoint:
    """Load policy from a checkpoint file."""

    checkpoint_path: Path | str = Path("runs/policy.pt")


@dataclass
class EvalFromConfig:
    """Create policy from config + dataset (for norm_stats)."""

    policy: PolicyConfig
    dataset: LeRobotDatasetConfig


EvalSource = (
    Annotated[EvalFromCheckpoint, tyro.conf.subcommand("checkpoint")]
    | Annotated[EvalFromConfig, tyro.conf.subcommand("config")]
)


@dataclass
class EvalConfig:
    """WebSocket policy server arguments."""

    source: EvalSource = field(default_factory=EvalFromCheckpoint)
    host: str = "0.0.0.0"
    port: int = 8765


def main(config: EvalConfig) -> None:
    if isinstance(config.source, EvalFromCheckpoint):
        policy = load_policy(config.source.checkpoint_path)
    else:
        dataset = make_dataset(config.source.dataset)
        policy = make_policy(config.source.policy, norm_stats=dataset.norm_stats)

    server = WebsocketPolicyServer(policy, host=config.host, port=config.port)
    server.serve_forever()


if __name__ == "__main__":
    main(tyro.cli(EvalConfig, config=(tyro.conf.CascadeSubcommandArgs,)))
