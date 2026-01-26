import tyro
from typing_extensions import Annotated

from vlla.models.base_model import BaseModel
from vlla.models.pi0 import Pi0ModelConfig
from vlla.models.pi05 import Pi05ModelConfig

ModelConfig = (
    Annotated[Pi05ModelConfig, tyro.conf.subcommand("pi05")]
    | Annotated[Pi0ModelConfig, tyro.conf.subcommand("pi0")]
)


def make_model(config: ModelConfig) -> BaseModel:
    """Create a model from config."""
    if isinstance(config, Pi05ModelConfig):
        from vlla.models.pi05 import Pi05

        return Pi05(config)
    elif isinstance(config, Pi0ModelConfig):
        from vlla.models.pi0 import Pi0

        return Pi0(config)
    raise ValueError(f"Unknown model config: {config}")


__all__ = [
    "ModelConfig",
    "make_model",
]
