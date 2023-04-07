import torch
from abc import ABC, abstractmethod


class Manifold(ABC):
    @abstractmethod
    def egrad2rgrad(
        self,
        point: torch.Tensor,
        egrad: torch.Tensor
    ) -> torch.Tensor:
        return egrad

    @abstractmethod
    def inner_product(
        self,
        point: torch.Tensor,
        tangent_vector1: torch.Tensor,
        tangent_vector2: torch.Tensor
    ) -> float:
        return tangent_vector1 @ tangent_vector2

    @abstractmethod
    def norm(
        self,
        point: torch.Tensor,
        tangent_vector: torch.Tensor,
    ) -> float:
        return torch.norm(tangent_vector)

    @abstractmethod
    def retraction(
        self,
        point: torch.Tensor,
        tangent_vector: torch.Tensor,
    ) -> torch.Tensor:
        return point + tangent_vector
    
    @abstractmethod
    def vector_transport(
        self,
        point: torch.Tensor,
        tangent_vector1: torch.Tensor,
        tangent_vector2: torch.Tensor,
        is_scaled: bool=False
    ) -> torch.Tensor:
        return tangent_vector2