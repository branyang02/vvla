from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import numpy as np


class BasePolicy(ABC):
    """Abstract base class for all policies."""

    @abstractmethod
    def infer(self, obs: dict) -> dict:
        """Given an observation dict, return an action dict."""
        ...

    def reset(self) -> None:
        """Reset any internal state."""
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save checkpoint with model weights + transforms."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> Self:
        """Load policy from checkpoint."""
        ...


class DummyPolicy(BasePolicy):
    """A dummy policy that returns random actions given an observation."""

    def __init__(self, action_dim: int = 8):
        self.action_dim = action_dim

    def infer(self, obs: dict) -> dict:
        return {"actions": np.random.randn(self.action_dim).astype(np.float32)}

    def save(self, path: str | Path) -> None:
        raise NotImplementedError("DummyPolicy does not support checkpointing")

    @classmethod
    def load(cls, path: str | Path) -> Self:
        raise NotImplementedError("DummyPolicy does not support checkpointing")
