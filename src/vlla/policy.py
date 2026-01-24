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


class MetaWorldDummyPolicy(BasePolicy):
    """A dummy policy for Meta-World that returns random actions for each env.
    https://metaworld.farama.org/benchmark/action_space/#

    Action space: Box(-1.0, 1.0, (4,), float32)
        [0]: dx - end-effector x displacement
        [1]: dy - end-effector y displacement
        [2]: dz - end-effector z displacement
        [3]: gripper control
    """

    def __init__(self, action_dim: int = 4):
        self.action_dim = action_dim

    def infer(self, obs: dict) -> dict:
        batch_size = obs["state"].shape[0]
        return {
            "actions": np.random.uniform(
                -1.0, 1.0, size=(batch_size, self.action_dim)
            ).astype(np.float32)
        }


class LiberoDummyPolicy(BasePolicy):
    def __init__(self, action_dim: int = 6, chunk_size: int = 5):
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    def infer(self, obs: dict) -> dict:
        # libero_client expects a chunk of actions
        # action_chunk = [[0.0] * 6 + [-1.0]] * self.chunk_size
        action_chunk = np.random.uniform(-1.0, 1.0, (self.chunk_size, 7))
        return {"actions": np.array(action_chunk, dtype=np.float32)}
