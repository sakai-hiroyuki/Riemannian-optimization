import torch

from manifolds import Manifold
from optimizers import Optmizer,  Linesearch


class SteepestDescent(Optmizer):
    def __init__(self, linesearch: Linesearch) -> None:
        self.linesearch = linesearch
    
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

        history = []

        for _ in range(max_iter):
            loss = f(point)
            loss.backward()

            egrad = point.grad
            rgrad = manifold.egrad2rgrad(point, egrad)

            descent_direction = -rgrad

            alpha = self.linesearch.search(f, manifold, point, rgrad, descent_direction)

            point = manifold.retraction(point, alpha * descent_direction)
            point = point.detach()
            point.requires_grad = True

            history.append(torch.norm(rgrad).item())
        
        return history