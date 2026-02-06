"""
scripts/eval.py

uv run python scripts/eval.py --help
"""

from dataclasses import dataclass

import tyro

from vvla.policies import PolicyConfig, make_policy
from vvla.serving.websocket_policy_server import WebsocketPolicyServer


@dataclass
class EvalConfig:
    """WebSocket policy server arguments."""

    policy: PolicyConfig
    host: str = "0.0.0.0"
    port: int = 8765


def main(config: EvalConfig) -> None:
    policy = make_policy(config.policy, mode="eval")

    server = WebsocketPolicyServer(policy, host=config.host, port=config.port)
    server.serve_forever()


if __name__ == "__main__":
    main(tyro.cli(EvalConfig, config=(tyro.conf.CascadeSubcommandArgs,)))
