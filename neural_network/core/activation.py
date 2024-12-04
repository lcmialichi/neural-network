from abc import ABC, abstractmethod
from typing import Union

class Activation(ABC):

    @abstractmethod
    def activate(self, x: float, alpha: Union[int, float, None] = None) -> Union[float, int]:
        pass
    
    @abstractmethod
    def derivate(self, x: float, alpha: Union[int, float, None] = None) :
        pass