from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from vvla.models.base_model import BaseModel
from vvla.policies.base_policy import BasePolicy, BasePolicyConfig
from vvla.transforms.transforms import DataTransformFn


@dataclass
class MetaworldPolicyConfig(BasePolicyConfig):
    action_dim: int = 4


class MetaworldPolicy(BasePolicy):
    """
    Policy for Metaworld environments wrapping model + transforms.

    https://metaworld.farama.org/benchmark/action_space/#

    Action space: Box(-1.0, 1.0, (4,), float32)
        [0]: dx - end-effector x displacement
        [1]: dy - end-effector y displacement
        [2]: dz - end-effector z displacement
        [3]: gripper control
    """

    policy_type = "metaworld"

    def __init__(
        self,
        config: MetaworldPolicyConfig,
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
        batch_size = obs["state"].shape[0]
        return {
            "actions": np.random.uniform(
                -1.0, 1.0, size=(batch_size, self.config.action_dim)
            ).astype(np.float32)
        }

    def train_forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """For training: data already transformed by dataset, just run model."""
        return self.model(batch)
