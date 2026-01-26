from dataclasses import dataclass

from torch import Tensor, nn

from vlla.models.base_model import BaseModel


@dataclass
class Pi0ModelConfig:
    input_dim: int = 39
    output_dim: int = 4
    hidden_dim: int = 128
    output_key: str = "actions"


class Pi0(BaseModel):
    """A simple MLP model for Pi0 environments."""

    def __init__(self, config: Pi0ModelConfig):
        super().__init__()
        self.config = config

        # A simple Dummy MLP: Linear -> ReLU -> Linear -> Output
        self.network = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        out = self.network(x["input"])
        return {self.config.output_key: out}
