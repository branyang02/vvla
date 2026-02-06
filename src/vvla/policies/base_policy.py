import tempfile
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Self

import safetensors
from huggingface_hub import HfApi, snapshot_download

from vvla.models import ModelConfig, make_model
from vvla.models.base_model import BaseModel
from vvla.models.pi05 import Pi05ModelConfig
from vvla.transforms import TransformsConfig
from vvla.transforms.pi05_transforms import Pi05TransformsConfig
from vvla.transforms.transforms import DataTransformFn


@dataclass
class BasePolicyConfig:
    """Base configuration for all policies.

    Contains common fields needed by any policy:
    - model: The neural network architecture config
    - transforms: Data processing transforms config
    - pretrained_repo_id_or_path: HuggingFace repo to load pretrained weights from
    - repo_id: Target HuggingFace repo for push_to_hub()

    Subclasses add robot/environment-specific fields like action_dim, chunk_size, etc.
    """

    model: ModelConfig = field(default_factory=Pi05ModelConfig)
    transforms: TransformsConfig = field(default_factory=Pi05TransformsConfig)

    # Target HuggingFace repo for push_to_hub(). Must be provided for training (e.g., "my-org/my-policy")
    repo_id: str | None = None
    # Optional HuggingFace repo ID to load pretrained weights from (e.g., "my-org/my-pretrained-policy").
    pretrained_repo_id_or_path: str | None = None


class BasePolicy(ABC):
    """Abstract base class for all policies."""

    policy_type: Literal["libero", "metaworld", "dummy"]
    config: BasePolicyConfig
    model: BaseModel
    input_transforms: Sequence[DataTransformFn]
    output_transforms: Sequence[DataTransformFn]

    @abstractmethod
    def infer(self, obs: dict) -> dict:
        """Given an observation dict, return an action dict."""
        ...

    def reset(self) -> None:
        """Reset any internal state."""
        pass

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Save policy weights and transforms.

        Creates:
            - model.safetensors: Model weights
            - transforms.json: Input/output transforms with NormStats

        Args:
            save_directory: Directory to save the policy files.
        """
        from vvla.policies import hub_utils

        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights in safetensors format
        safetensors.torch.save_file(
            self.model.state_dict(), save_dir / "model.safetensors"
        )

        # Save transforms
        hub_utils.save_transforms_json(
            input_transforms=list(self.input_transforms),
            output_transforms=list(self.output_transforms),
            path=save_dir / "transforms.json",
        )

    @classmethod
    def from_config(
        cls,
        config: BasePolicyConfig,
        *,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        token: str | bool | None = None,
    ) -> Self:
        from vvla.policies import hub_utils

        if config.pretrained_repo_id_or_path is None:
            raise ValueError(
                "`config.pretrained_repo_id_or_path` is required to load a checkpoint."
            )

        # Check if it's a local path or HF repo
        path = Path(config.pretrained_repo_id_or_path)
        if path.exists() and path.is_dir():
            local_dir = path
        else:
            # Download from HuggingFace Hub
            local_dir = Path(
                snapshot_download(
                    repo_id=str(config.pretrained_repo_id_or_path),
                    revision=revision,
                    cache_dir=cache_dir,
                    token=token,
                    allow_patterns=[
                        "model.safetensors",
                        "transforms.json",
                    ],
                )
            )

        # Load model weights
        model = make_model(config.model)
        state_dict = safetensors.torch.load_file(local_dir / "model.safetensors")
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to load weights from {local_dir / 'model.safetensors'}: {e}"
            ) from e

        # Load transforms
        input_transforms, output_transforms = hub_utils.load_transforms_json(
            local_dir / "transforms.json"
        )

        return cls(
            config=config,
            model=model,
            input_transforms=input_transforms,
            output_transforms=output_transforms,
        )

    def push_to_hub(
        self,
        *,
        commit_message: str | None = None,
        private: bool = False,
        token: str | bool | None = None,
        create_pr: bool = False,
    ) -> str:
        """Push policy to HuggingFace Hub.

        Args:
            commit_message: Git commit message for the upload.
            private: Whether to create a private repository.
            token: HF token for authentication. True uses cached token.
            create_pr: If True, creates a PR instead of direct push.

        Returns:
            URL of the uploaded repository or pull request.
        """

        api = HfApi(token=token)

        # Create repo if it doesn't exist
        api.create_repo(repo_id=self.config.repo_id, private=private, exist_ok=True)

        # Save to temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            self.save_pretrained(tmpdir)

            # Upload all files
            return api.upload_large_folder(
                folder_path=tmpdir,
                repo_id=self.config.repo_id,
                repo_type="model",
                commit_message=commit_message or f"Upload {self.policy_type} policy",
                create_pr=create_pr,
            )
