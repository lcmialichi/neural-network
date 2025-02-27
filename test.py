import numpy as np
from neural_network.core import Initialization
from neural_network.core.padding import Padding
from neural_network.blocks.block import Block
from neural_network.blocks.kernel import Kernel
from neural_network.configuration import Driver, GlobalConfig
from neural_network import Config
import neural_network.supply as attr
from neural_network.core.image_processor import ImageProcessor

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
        return np.array([0.1, 0.2], dtype=np.float32)

    def generate_layer(self, input_size: int, size: int) -> list:
        return np.array(
            [[-0.19436161,  0.6982436 ],
            [ 0.35940347,  0.15284107],
            [-0.53289363, -0.532931  ],
            [-0.68461392,  0.56727765],
            [ 0.1566467,   0.32234465],
            [-0.74270731,  0.72798121],
            [ 0.51501792, -0.44564233],
            [-0.49291464, -0.49046762]]
            , dtype=np.float32)
    
  
def test_forward_pass():
    input_layer = np.array([[
        [[1.0], [2.0], [3.0], [4.0]],
        [[5.0], [6.0], [7.0], [8.0]],
        [[9.0], [10.0], [11.0], [12.0]],
        [[13.0], [14.0], [15.0], [16.0]]
    ]], dtype=np.float32)

    config = Config()
    config.padding_type(Padding.SAME)
    
    config.set_processor(
        ImageProcessor(
            base_dir="/content/neural-network/data/breast-histopathology-images/IDC_regular_ps50_idx5",
            image_size=(50, 50),
            batch_size=32,
            split_ratios=(0.90, 0.10),
            shuffle=True,
            augmentation=True,
            augmentation_params={
                'rotation': 20,
                'zoom': 0.2,
                'horizontal_flip': True,
                'shear': 0.2,
                'fill_mode': 'nearest'
            }
        )
    )
    
    config.loss_function(attr.CrossEntropyLoss())

    kernel = Kernel(number=2, shape=(2, 2), stride=1, bias=False)
    kernel.initializer(MockInitializer())
    kernel.max_pooling(shape=(2, 2), stride=2)
    kernel.padding_type = Padding.SAME
    config.add(kernel)
    
    flatten = config.flatten()
    
    dense = config.dense().add_layer(size=2)
    dense.activation(attr.Softmax())
    dense.initializer(MockInitializer())
    
    optimer = attr.Adam(
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=False,
        amsgrad=False
    )
    
    kernel.global_optimizer = optimer
    dense.global_optimizer = optimer
    
    output_kernel = kernel.forward(input_layer)
    print("kernel output:")
    print(output_kernel)
    output_flatten = flatten.forward(output_kernel)
    print("flatten output:")
    print(output_flatten)
    print("Output:")
    output_dense = dense.forward(output_flatten)
    print(output_dense)
    
    y_true = np.array([[1, 0]], dtype=np.float32)
    gradient = y_true - output_dense
    print("Gradient:")
    print(gradient)
    back_dense_output = dense.backward(output_flatten, y_true, gradient)
    
    print('backpropagation dense output:')
    print(back_dense_output)
    
    back_flatten_output = flatten.backward(output_kernel, y_true, back_dense_output)
    print('backpropagation flatten output:')
    print(back_flatten_output)
    
    back_kernel_output = kernel.backward(input_layer, y_true, back_flatten_output)
    print('backpropagation kernel output:')
    print(back_kernel_output)
    

GlobalConfig().set_driver(Driver['cpu'])
test_forward_pass()
