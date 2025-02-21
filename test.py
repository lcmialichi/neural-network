import numpy as np
from neural_network.gcpu import driver
from neural_network.core import Initialization
from neural_network.core.padding import Padding
from neural_network.support import conv
from neural_network.blocks.block import Block
from neural_network.blocks.kernel import Kernel
from neural_network.configuration import Driver, GlobalConfig


import uuid

class MockInitializer(Initialization):
    
    def kernel_filters(self, filter_number: int, filter_shape: tuple[int, int], channels_number: int):
        return np.array([
            [[[0.19, -0.01]],
            [[0.99, 1.21]]
        ],[
            [[0.90, -1.10 ]],
            [[0.73, -0.27]]
        ]], dtype=np.float32)

    def kernel_bias(self, number: int):
        return np.array([0.1, 0.2], dtype=np.float32)

    def generate_layer_bias(self, size: int) -> list:
        return driver.gcpu.zeros(size)

    def generate_layer(self, input_size: int, size: int) -> list:
        pass
    
  
def test_forward_pass():
    input_layer = np.array([[
        [[1.0], [2.0], [3.0], [4.0]],
        [[5.0], [6.0], [7.0], [8.0]],
        [[9.0], [10.0], [11.0], [12.0]],
        [[13.0], [14.0], [15.0], [16.0]]
    ]], dtype=np.float32)

    expected_output = np.array([[
        [[11.178139,  -4.5924845], 
         [16.517935,   4.4727635], 
         [21.85773,   13.538012 ], 
         [13.358887,   9.506775 ]],
        [
         [24.628832,  -4.990206 ], 
         [34.39431,    6.476242 ], 
         [44.159782,  17.94269  ], 
         [25.67662,   13.599579 ]],
        [
         [38.079525,  -5.387928 ], 
         [52.27068,    8.479721 ], 
         [66.46184,   22.34737  ], 
         [37.99435,   17.692383 ]],
        [
         [24.418777,  -3.8382874], 
         [33.518127,   5.158844 ], 
         [42.617477,  14.155975 ], 
         [24.57284,   10.927643 ]]
    ]], dtype=np.float32)


    kernel = Kernel(number=2, shape=(2, 2), stride=1, bias=False)
    kernel.initializer(MockInitializer())
    kernel.padding_type = Padding.SAME

    output = kernel.forward(input_layer)
    print(output)


GlobalConfig().set_driver(Driver['cpu'])
test_forward_pass()
