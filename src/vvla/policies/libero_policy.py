from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from vvla.models.base_model import BaseModel
from vvla.policies.base_policy import BasePolicy, BasePolicyConfig
from vvla.transforms.transforms import DataTransformFn


@dataclass
class LiberoPolicyConfig(BasePolicyConfig):
    action_dim: int = 7
    chunk_size: int = 5


class LiberoPolicy(BasePolicy):
    """Policy for Libero environments wrapping model + transforms."""

    policy_type = "libero"

    def __init__(
        self,
        config: LiberoPolicyConfig,
        model: BaseModel,
        input_transforms: Sequence[DataTransformFn] = (),
        output_transforms: Sequence[DataTransformFn] = (),
    ):
        self.config = config
        self.model = model
        self.input_transforms = input_transforms
        self.output_transforms = output_transforms

    @torch.no_grad()
    def infer(self, obs: dict) -> dict:
        # libero_client expects a chunk of actions
        action_chunk = np.random.uniform(
            -1.0, 1.0, (self.config.chunk_size, self.config.action_dim)
        )
        return {"actions": np.array(action_chunk, dtype=np.float32)}

    def train_forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """For training: data already transformed by dataset, just run model."""
        return self.model(batch)
