from typing import Union
from neural_network.blocks import Layer
from neural_network.blocks.block import Block
from .propagable import Propagable

class Dense(Propagable):
    _block_output: list = []
    global_optimizer = None

    def __init__(self):
        self._optimizer = None
        self._blocks: list[Block] = []
        self.mode = None
        self.regularization_lambda = None
        self.loss_function = None

    def add_layer(
        self, 
        size: int, 
        dropout: Union[float, None] = None,
        ) -> "Layer":
        layer = Layer(size=size, dropout=dropout)
        self._blocks.append(layer)
        return layer

    def forward(self, x):
        self._block_output.clear()
        output = x
        for block in self._blocks:
            if hasattr(block, 'mode'):
                block.mode = self.mode
            if hasattr(block, 'regularization_lambda'):
                block.regularization_lambda = self.regularization_lambda
            if hasattr(block, 'global_optimizer'):
                block.global_optimizer = self.global_optimizer
                
            output = block.forward(output)
            self._block_output.append(output)

        return output

    def backward(self, input_layer, y, output):
        delta = self.loss_function.gradient(output, y)
        for i in range(len(self._blocks) - 1, -1, -1):
            block = self._blocks[i]
            prev_output = self._block_output[i - 1] if i > 0 else input_layer
            delta = block.backward(prev_output, y, delta)
        return delta.reshape(input_layer.shape)
    
    def boot(self, shape: tuple):
        return
