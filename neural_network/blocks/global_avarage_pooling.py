from .propagable import Propagable
from neural_network.gcpu import driver

class GlobalAveragePooling(Propagable):
    def forward(self, x):
        self.input_shape = x.shape
        return driver.gcpu.mean(x, axis=(2, 3))
    
    def backward(self, input_layer, y, delta):
        _, _, height, width = self.input_shape
        delta_expanded = delta[:, :, driver.gcpu.newaxis, driver.gcpu.newaxis]
        return driver.gcpu.ones(self.input_shape) * (delta_expanded / (height * width))
    
    def boot(self, shape: tuple):
        return
