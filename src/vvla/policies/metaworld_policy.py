from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np
import torch
from torch import Tensor

from vvla.models import ModelConfig
from vvla.models.base_model import BaseModel
from vvla.models.pi05 import Pi05ModelConfig
from vvla.policies.base_policy import BasePolicy
from vvla.transforms import TransformsConfig
from vvla.transforms.pi05_transforms import Pi05TransformsConfig
from vvla.transforms.transforms import DataTransformFn


@dataclass
class MetaworldPolicyConfig:
    model: ModelConfig = field(default_factory=Pi05ModelConfig)
    transforms: TransformsConfig = field(default_factory=Pi05TransformsConfig)

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

    def save(self, path: str | Path) -> None:
        """Save checkpoint with model weights + transforms."""
        checkpoint = {
            "policy_type": "metaworld",
            "config": self.config,
            "model_state_dict": self.model.state_dict(),
            "input_transforms": list(self.input_transforms),
            "output_transforms": list(self.output_transforms),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load policy from checkpoint."""
        from vvla.models import make_model

        checkpoint = torch.load(path, weights_only=False)
        model = make_model(checkpoint["config"].model)
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(
            config=checkpoint["config"],
            model=model,
            input_transforms=checkpoint["input_transforms"],
            output_transforms=checkpoint["output_transforms"],
        )
