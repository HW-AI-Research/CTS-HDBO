# from baxus.benchmarks import EffectiveDimBoTorchBenchmark
# from botorch.test_functions import Branin as BotorchBranin
from botorch.test_functions.multi_objective import (
    BraninCurrin,
)
from typing import Optional
from torch import Tensor
import torch

class BraninCurrinEffectiveDim(BraninCurrin):

    def __init__(
            self, dim: int = 200, noise_std: Optional[float] = None, negate: bool = False
    ):
        super().__init__(
            noise_std=noise_std,
            negate=negate
        )
        self.bounds = torch.cat([torch.zeros((1, dim)), torch.ones((1, dim))], dim=0)
    
    def evaluate_true(self, X: Tensor) -> Tensor:
        return super().evaluate_true(X[:, :2])