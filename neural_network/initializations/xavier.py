from neural_network.core import Initialization
from neural_network.gcpu import gcpu

class Xavier(Initialization):
    
    def generate_layer(self, input_size: int, size: int) -> list:
        generator = gcpu.random.default_rng(42)
        stddev = gcpu.sqrt(2 / (input_size + size))
        return generator.normal(0, stddev, (input_size, size))



