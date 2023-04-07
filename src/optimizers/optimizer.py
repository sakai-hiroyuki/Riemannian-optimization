import torch
from abc import ABC, abstractmethod


class Optmizer(ABC):
    @abstractmethod
    def solve(self) -> None:
        ...
