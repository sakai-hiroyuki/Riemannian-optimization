import torch
from math import sqrt

from manifolds import Manifold, Sphere


m = Sphere(3)
x = torch.Tensor([0, 0, 1])
v = torch.Tensor([1, 0, 0])
u = torch.Tensor([1, 1, 0])

t = m.vector_transport(x, v, u, is_scaled=False)
print(t)


print(torch.Tensor([1, 2, -1]) / sqrt(2) / 2)
