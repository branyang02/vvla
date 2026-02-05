from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self


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
