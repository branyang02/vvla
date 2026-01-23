"""
scripts/eval.py

Evaluation server that serves a dummy policy over WebSocket.
"""

from dataclasses import dataclass

import tyro

from vlla.policy import DummyPolicy
from vlla.serving.websocket_policy_server import WebsocketPolicyServer


@dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8765
    action_dim: int = 8


def main(args: Args) -> None:
    policy = DummyPolicy(action_dim=args.action_dim)
    server = WebsocketPolicyServer(policy, host=args.host, port=args.port)
    server.serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
