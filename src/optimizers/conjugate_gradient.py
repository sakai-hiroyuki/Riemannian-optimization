import torch

from manifolds import Manifold
from optimizers import Optmizer,  Linesearch


class ConjugateGradient(Optmizer):
    def __init__(self, linesearch: Linesearch, betatype: str='FR') -> None:
        self.linesearch = linesearch
        self.betatype = betatype
    
    def solve(
        self,
        f,
        manifold: Manifold,
        initial_point: torch.Tensor,
        max_iter: int=1000,
        stop_condition: float=1e-6
    ) -> None:
        
        point = initial_point.detach()
        point.requires_grad = True

        loss = f(point)
        loss.backward()
        egrad = point.grad
        rgrad = manifold.egrad2rgrad(point, egrad)

        descent_direction = -rgrad

        history = []

        for _ in range(max_iter):
            history.append(torch.norm(rgrad).item())
            
            alpha = self.linesearch.search(f, manifold, point, rgrad, descent_direction)

            point_next = manifold.retraction(point, alpha * descent_direction)
            point_next = point_next.detach()
            point_next.requires_grad = True

            loss = f(point_next)
            loss.backward()

            egrad_next = point_next.grad
            rgrad_next = manifold.egrad2rgrad(point_next, egrad_next)

            beta = _compute_FR(manifold, point, rgrad, point_next, rgrad_next)

            descent_direction = -rgrad_next + beta * manifold.vector_transport(point, alpha * descent_direction, descent_direction)

            rgrad = rgrad_next
            point = point_next
        return history


def _compute_FR(
    manifold: Manifold,
    point: torch.Tensor,
    rgrad: torch.Tensor,
    point_next: torch.Tensor,
    rgrad_next: torch.Tensor
) -> torch.Tensor:
    _numer = manifold.inner_product(point_next, rgrad_next, rgrad_next)
    _denom = manifold.inner_product(point, rgrad, rgrad)
    return _numer / _denom
