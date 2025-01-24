from neural_network.core import Initialization
from neural_network.gcpu import gcpu

class He(Initialization):
    def generate_layer(self, input_size: int, size: int) -> list:
        generator = np.random
        stddev = gcpu.sqrt(2 / input_size)
        return generator.normal(0, stddev, size=(input_size, size))

        
