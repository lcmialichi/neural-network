from Activation import Activation
import numpy as np
from typing import Union

class Sigmoid(Activation):
    def activate(self, x: float, alpha: Union[int, float, None] = None) -> Union[float, int]:
        return 1 / (1 + np.exp(-x))

    def derivate(self, x: float, alpha: Union[int, float, None] = None) -> Union[float, int]:
        return x * (1 - x)