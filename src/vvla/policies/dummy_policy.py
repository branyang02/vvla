from dataclasses import dataclass

import numpy as np

from vvla.policies.base_policy import BasePolicy


@dataclass
class DummyPolicyConfig:
    action_dim: int = 7
    chunk_size: int = 5


class DummyPolicy(BasePolicy):
    policy_type = "dummy"

    def __init__(self, config: DummyPolicyConfig):
        self.config = config

    def infer(self, obs: dict) -> dict:
        return {"actions": np.random.randn(self.config.action_dim).astype(np.float32)}
