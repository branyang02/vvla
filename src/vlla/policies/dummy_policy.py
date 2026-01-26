from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np

from vlla.policies.base_policy import BasePolicy


@dataclass
class DummyPolicyConfig:
    action_dim: int = 7
    chunk_size: int = 5


class DummyPolicy(BasePolicy):
    def __init__(self, config: DummyPolicyConfig):
        self.config = config

    def infer(self, obs: dict) -> dict:
        return {"actions": np.random.randn(self.config.action_dim).astype(np.float32)}

    def reset(self) -> None:
        pass

    def save(self, path: str | Path) -> None:
        pass

    @classmethod
    def load(cls, path: str | Path) -> Self:
        return cls(DummyPolicyConfig())
