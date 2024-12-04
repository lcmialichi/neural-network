import Activation
import numpy as np
from typing import Union

class LeakyRelu(Activation):
    def activate(self, x: float, alpha: Union[int, float, None] = 0.1) -> Union[float, int]:
        return np.where(x > 0, x, x * alpha)

    def derivate(self, x: float, alpha: Union[int, float, None] = 0.1) -> Union[float, int]:
        return (x > 0).astype(float)
