"""
scripts/eval.py

uv run python scripts/eval.py --help
"""

from dataclasses import dataclass, field

import tyro

from vlla.policy import BasePolicy, DummyPolicy, LiberoDummyPolicy, MetaWorldDummyPolicy
from vlla.serving.websocket_policy_server import WebsocketPolicyServer


### See https://brentyi.github.io/tyro/examples/subcommands/ for subcommand examples. ###
@dataclass
class Dummy:
    action_dim: int = 8


@dataclass
class MetaWorld:
    action_dim: int = 4


@dataclass
class Libero:
    action_dim: int = 6
    chunk_size: int = 5


def make_policy(policy_config: Dummy | MetaWorld | Libero) -> BasePolicy:
    if isinstance(policy_config, Dummy):
        return DummyPolicy(action_dim=policy_config.action_dim)
    elif isinstance(policy_config, MetaWorld):
        return MetaWorldDummyPolicy(action_dim=policy_config.action_dim)
    elif isinstance(policy_config, Libero):
        return LiberoDummyPolicy(
            action_dim=policy_config.action_dim, chunk_size=policy_config.chunk_size
        )
    else:
        raise ValueError(f"Unknown policy config: {policy_config}")


@dataclass
class Args:
    policy: Dummy | MetaWorld | Libero = field(default_factory=Dummy)
    host: str = "0.0.0.0"
    port: int = 8765


def main(args: Args) -> None:
    policy = make_policy(args.policy)
    server = WebsocketPolicyServer(policy, host=args.host, port=args.port)
    server.serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
