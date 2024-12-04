from Activation import Activation
import numpy as np
from typing import Union

class Relu(Activation):
    def activate(self, x: float, alpha: Union[int, float, None] = None) -> Union[float, int]:
        return np.maximum(0, x)

    def derivate(self, x: float, alpha: Union[int, float, None] = None) -> Union[float, int]:
        return (x > 0).astype(float)
