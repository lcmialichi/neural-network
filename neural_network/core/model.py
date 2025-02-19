from neural_network.core.base_network import BaseNetwork
from neural_network.train import CnnTrainer
from neural_network.storage import Storage
from typing import Union
from neural_network.supply.loss import LossFunction
from neural_network.blocks.block import Propagable
from neural_network.gcpu import driver

class Model(BaseNetwork):

    _mode: str = 'test'
    
    def __init__(self, config: dict, storage: Union[None, Storage]):
        if 'processor' not in config:
            raise ValueError('processor not defined in configuration')
        
        self._storage = storage
        self._processor = config.get('processor')
        self._block_output: list = []
        self._padding_type = config.get('padding_type')
        self._regularization_lambda = config.get('regularization_lambda')
        self._global_optimizer = config.get('global_optimizer')
        self._blocks: list[Propagable] = config.get('blocks', [])
        self._loss_function = config.get('loss_function')
        self._num_predictions = 0
        driver.gcpu.random.seed(42)
        
    def forward(self, x):
        self._block_output.clear()
        output = x
        for block in self._blocks:
            self.set_hyper_params(block)
            if self._num_predictions == 0:
                self.boot_block(block, output.shape)

            output = block.forward(output)
            self._block_output.append(output)

        self._num_predictions += 1
        return output

    def backward(self, x, y, output):
        delta_conv = output
        for i in range(len(self._blocks) - 1, -1, -1):
            block = self._blocks[i]
            input_layer = x if i == 0 else self._block_output[i - 1]
            delta_conv = block.backward(input_layer,y, delta_conv)
        return delta_conv

    def train(self, x_batch, y_batch):
        output_batch = self.forward(x_batch)
        self.backward(x_batch, y_batch, output_batch)
        return output_batch

    def save_state(self):
        if self._storage:
            self._storage.store(self)

    def boot_block(self, block, shape: tuple):
        if hasattr(block, 'boot'):
            block.boot(shape)

    def predict(self, x):
        return self.forward(x)

    def get_trainer(self):
        return CnnTrainer(self, self.get_processor())
    
    def get_loss_function(self) -> "LossFunction":
        return self._loss_function
    
    def has_loss_function(self) -> bool:
        return self._loss_function is not None
    
    def set_hyper_params(self, block: "Propagable"):
        if hasattr(block, 'padding_type'):
            block.padding_type = self._padding_type
        if hasattr(block, 'mode'):
            block.mode = self._mode
        if hasattr(block, 'regularization_lambda'):
            block.regularization_lambda = self._regularization_lambda
        if hasattr(block, 'global_optimizer'):
            block.global_optimizer = self._global_optimizer
        if hasattr(block, 'loss_function'):
            block.loss_function = self._loss_function