from dataclasses import dataclass

import numpy as np
import tyro
from loguru import logger

from websocket_policy_client import WebsocketPolicyClient


@dataclass
class Args:
    host: str = "localhost"
    port: int = 8765
    obs_dim: int = 10
    num_steps: int = 5


def random_obs() -> dict:
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "prompt": "do something",
    }


def main(args: Args) -> None:
    policy = WebsocketPolicyClient(host=args.host, port=args.port)
    logger.info(f"Server metadata: {policy.server_metadata}")

    for step in range(args.num_steps):
        obs = random_obs()
        result = policy.infer(obs)
        actions = result["actions"]
        logger.info(f"Step {step + 1}/{args.num_steps}: Received actions {actions}")


if __name__ == "__main__":
    main(tyro.cli(Args))
