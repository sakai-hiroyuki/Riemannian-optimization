import torch
from abc import ABC, abstractmethod

from manifolds import Manifold


class Linesearch(ABC):
    @abstractmethod
    def search(
        self,
        f,
        manifold: Manifold,
        point: torch.Tensor,
        rgrad: torch.Tensor,
        descent_direction: torch.Tensor,
    ) -> float:
        ...


class LinesearchArmijo(Linesearch):
    def __init__(self, c1: float=1e-4) -> None:
        if not 0.0 < c1 < 1.0:
            raise ValueError(f'Invalid value: c1={c1}.')
        self.c1 = c1

    def search(
        self,
        f,
        manifold: Manifold,
        point: torch.Tensor,
        rgrad: torch.Tensor,
        descent_direction: torch.Tensor,
    ) -> float:
        
        def phi(alpha) -> float:
            return f(manifold.retraction(point, alpha * descent_direction))
        
        step = 1.
        while phi(step) > phi(0.) + self.c1 * step * manifold.inner_product(point, rgrad, descent_direction):
            step *= 0.9
        return step