from typing import Annotated, Literal

import tyro
from loguru import logger

from vvla.models import make_model
from vvla.policies.base_policy import BasePolicy
from vvla.policies.dummy_policy import DummyPolicy, DummyPolicyConfig
from vvla.policies.libero_policy import LiberoPolicy, LiberoPolicyConfig
from vvla.policies.metaworld_policy import MetaworldPolicy, MetaworldPolicyConfig
from vvla.transforms import make_transforms
from vvla.transforms.transforms import NormStats

PolicyConfig = (
    Annotated[LiberoPolicyConfig, tyro.conf.subcommand("libero")]
    | Annotated[MetaworldPolicyConfig, tyro.conf.subcommand("metaworld")]
    | Annotated[DummyPolicyConfig, tyro.conf.subcommand("dummy")]
)


def make_policy(
    config: PolicyConfig,
    *,
    mode: Literal["train", "eval"],
    norm_stats: dict[str, NormStats] | None = None,
) -> BasePolicy:
    if isinstance(config, DummyPolicyConfig):
        return DummyPolicy(config=config)

    _validate_policy_inputs(config=config, mode=mode, norm_stats=norm_stats)

    # OPTION 1: Finetune from pretrained model
    # This is also the ONLY option for eval, since eval always loads from a checkpoint.
    if config.pretrained_repo_id_or_path is not None:
        if isinstance(config, LiberoPolicyConfig):
            policy = LiberoPolicy.from_config(config)
        elif isinstance(config, MetaworldPolicyConfig):
            policy = MetaworldPolicy.from_config(config)
        else:
            raise ValueError(f"Unknown policy config type: {type(config)}")

        if mode == "train":
            assert norm_stats is not None
            input_transforms, output_transforms = make_transforms(
                config.transforms, norm_stats
            )
            # TODO(branyang02): Allow fine-tuning to optionally reuse checkpoint transforms.
            # Current behavior: always rebuild transforms (incl. norm stats) from the dataset and
            # overwrite the checkpoint transforms. This is OK for now, but redundant when
            # fine-tuning on the same dataset and may break exact reproducibility.
            policy.input_transforms = input_transforms
            policy.output_transforms = output_transforms

        return policy

    # OPTION 2: Create fresh policy and train from scratch
    logger.info(
        "Creating a new policy with randomly initialized weights (no pretrained checkpoint). "
        "This is intended for training from scratch. If you meant to load a pretrained model, "
        "please set `config.pretrained_repo_id_or_path` to the appropriate Hugging Face repo or "
        "local checkpoint path."
    )

    model = make_model(config.model)
    assert norm_stats is not None
    input_transforms, output_transforms = make_transforms(config.transforms, norm_stats)

    if isinstance(config, LiberoPolicyConfig):
        return LiberoPolicy(
            config=config,
            model=model,
            input_transforms=input_transforms,
            output_transforms=output_transforms,
        )
    elif isinstance(config, MetaworldPolicyConfig):
        return MetaworldPolicy(
            config=config,
            model=model,
            input_transforms=input_transforms,
            output_transforms=output_transforms,
        )
    raise ValueError(f"Unknown policy config: {config}")


def _validate_policy_inputs(
    config: PolicyConfig,
    mode: Literal["train", "eval"],
    norm_stats: dict[str, NormStats] | None,
) -> None:
    if mode not in {"train", "eval"}:
        raise ValueError(f"`mode` must be 'train' or 'eval', got: {mode!r}")

    if mode == "train":
        if config.repo_id is None:
            raise ValueError(
                "Training requires `config.repo_id` (used for save_pretrained()/push_to_hub()).\n"
                "Fix: set `--policy.repo_id my-org/my-policy`."
            )

        if norm_stats is None:
            raise ValueError(
                "Training requires `norm_stats` (dataset normalization stats), even when fine-tuning.\n"
                "Reason: training builds dataset transforms and needs normalization parameters.\n"
                "Fix: compute and pass norm_stats, e.g.\n"
                "  dataset = make_dataset(config.dataset)\n"
                "  policy = make_policy(config.policy, mode='train', norm_stats=dataset.norm_stats)\n"
            )

    elif mode == "eval":
        if config.repo_id is not None:
            raise ValueError(
                "Evaluation should not set `config.repo_id`.\n"
                "Reason: eval does not save/push artifacts; repo_id is training-only metadata.\n"
                "Fix: unset `--policy.repo_id` for eval configs."
            )

        if norm_stats is not None:
            raise ValueError(
                "Evaluation should not pass `norm_stats`.\n"
                "Reason: eval loads transforms/normalization from the checkpoint.\n"
                "Fix: call make_policy(..., mode='eval', norm_stats=None)."
            )

        if config.pretrained_repo_id_or_path is None:
            raise ValueError(
                "Evaluation requires `config.pretrained_repo_id_or_path` to load a checkpoint.\n"
                "Fix: set it to a local path or Hugging Face repo id, e.g.\n"
                "  `--policy.pretrained_repo_id_or_path my-org/my-policy` \n"
            )


__all__ = [
    "PolicyConfig",
    "make_policy",
]
