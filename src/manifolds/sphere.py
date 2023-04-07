import torch
from math import sin, cos

from manifolds import Manifold


class Sphere(Manifold):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n
    
    @property
    def dim(self) -> int:
        return self.n - 1
    
    def egrad2rgrad(
        self,
        point: torch.Tensor,
        egrad: torch.Tensor
    ) -> torch.Tensor:
        _v: torch.Tensor = point.reshape(self.n, 1)
        _proj = torch.eye(self.n) - _v @ _v.permute(1, 0)
        return _proj @ egrad
    
    def inner_product(
        self,
        point: torch.Tensor,
        tangent_vector1: torch.Tensor,
        tangent_vector2: torch.Tensor
    ) -> float:
        return super().inner_product(point, tangent_vector1, tangent_vector2)
    
    def norm(
        self,
        point: torch.Tensor,
        tangent_vector: torch.Tensor
    ) -> float:
        return super().norm(point, tangent_vector)
    
    def retraction(
        self,
        point: torch.Tensor,
        tangent_vector: torch.Tensor
    ) -> torch.Tensor:
        _numer: torch.Tensor = point + tangent_vector
        _denom: torch.Tensor = torch.norm(point + tangent_vector)
        return _numer / _denom
    
    def exponantial_map(
        self,
        point: torch.Tensor,
        tangent_vector: torch.Tensor
    ) -> torch.Tensor:
        _k: torch.Tensor = torch.norm(tangent_vector)
        return cos(_k) * point + sin(_k) * tangent_vector / _k

    def vector_transport(
        self,
        point: torch.Tensor,
        tangent_vector1: torch.Tensor,
        tangent_vector2: torch.Tensor,
        is_scaled: bool=False
    ) -> torch.Tensor:
        _v = point + tangent_vector1
        _p = _v.reshape(self.n, 1) / torch.norm(_v)
        _proj = torch.eye(self.n) - _p @ _p.permute(1, 0)
        _res = _proj @ tangent_vector2 / torch.norm(_v)

        scale: float = 1.
        if is_scaled:
            scale = min(1., torch.norm(tangent_vector1) / torch.norm(_res))

        return _res * scale