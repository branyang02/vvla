from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np
import torch
from torch import Tensor

from vlla.models import ModelConfig
from vlla.models.base_model import BaseModel
from vlla.models.pi05 import Pi05ModelConfig
from vlla.policies.base_policy import BasePolicy
from vlla.transforms import TransformsConfig
from vlla.transforms.pi05_transforms import Pi05TransformsConfig
from vlla.transforms.transforms import DataTransformFn


@dataclass
class LiberoPolicyConfig:
    model: ModelConfig = field(default_factory=Pi05ModelConfig)
    transforms: TransformsConfig = field(default_factory=Pi05TransformsConfig)

    action_dim: int = 7
    chunk_size: int = 5


class LiberoPolicy(BasePolicy):
    """Policy for Libero environments wrapping model + transforms."""

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

    def save(self, path: str | Path) -> None:
        """Save checkpoint with model weights + transforms."""
        checkpoint = {
            "policy_type": "libero",
            "config": self.config,
            "model_state_dict": self.model.state_dict(),
            "input_transforms": list(self.input_transforms),
            "output_transforms": list(self.output_transforms),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load policy from checkpoint."""
        from vlla.models import make_model

        checkpoint = torch.load(path, weights_only=False)
        model = make_model(checkpoint["config"].model)
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(
            config=checkpoint["config"],
            model=model,
            input_transforms=checkpoint["input_transforms"],
            output_transforms=checkpoint["output_transforms"],
        )
