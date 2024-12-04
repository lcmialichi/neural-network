from abc import ABC
from typing import Union
import numpy as np

class Initialization(ABC):
    
    @staticmethod
    def variance() -> Union[int, float]:
        return 1
    
    def generate_layers(
        self, 
        input_size: int, 
        output_size: int, 
        hidden_size: int, 
        layers_number: int
    )-> list:
        generator = np.random.default_rng(42)
        weights = []
        for _ in range(layers_number - 1):
            weights.append(generator.normal(0, np.sqrt(self.variance() / input_size), (input_size, hidden_size)))
            input_size = hidden_size

        weights.append(generator.normal(0, np.sqrt(self.variance() / hidden_size), (hidden_size, output_size)))
        return weights
    

