from abc import ABC, abstractmethod

import numpy as np


class BasePolicy(ABC):
    @abstractmethod
    def infer(self, obs: dict) -> dict:
        """Given an observation dict, return an action dict."""
        ...

    def reset(self) -> None:
        pass


class DummyPolicy(BasePolicy):
    """A dummy policy that returns random actions given an observation."""

    def __init__(self, action_dim: int = 8):
        self.action_dim = action_dim

    def infer(self, obs: dict) -> dict:
        return {"actions": np.random.randn(self.action_dim).astype(np.float32)}
