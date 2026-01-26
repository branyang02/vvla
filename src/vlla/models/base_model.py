from abc import abstractmethod

from torch import Tensor, nn


class BaseModel(nn.Module):
    """Base model class. All models take dict[str, Tensor] and return dict[str, Tensor]."""

    @abstractmethod
    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]: ...
