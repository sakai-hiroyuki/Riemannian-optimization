import torch
from math import sqrt
from sklearn.datasets import make_spd_matrix
from matplotlib import pyplot as plt

from manifolds import Sphere
from optimizers import SteepestDescent, ConjugateGradient, LinesearchArmijo


def f(x: torch.Tensor) -> torch.Tensor:
    return x @ A @ x


n = 100
manifold = Sphere(n)
A = torch.Tensor(make_spd_matrix(n))
# A = torch.Tensor(torch.diag(torch.Tensor([k + 1 for k in range(n)])))
point = torch.ones(n) / sqrt(n)
point.requires_grad = True

linesearch = LinesearchArmijo()
optimizer = SteepestDescent(linesearch)

y = optimizer.solve(f, manifold, initial_point=point)
plt.plot(range(1000), y)

linesearch = LinesearchArmijo()
optimizer = ConjugateGradient(linesearch)

y = optimizer.solve(f, manifold, initial_point=point)
plt.plot(range(1000), y)

plt.yscale('log')
plt.show()
