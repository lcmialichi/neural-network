from .propagable import Propagable
import numpy as np

class GlobalAveragePooling(Propagable):
    def forward(self, x):
        self.input_shape = x.shape
        return np.mean(x, axis=(2, 3))
    
    def backward(self, input_layer, y, delta):
        _, _, height, width = self.input_shape
        delta_expanded = delta[:, :, np.newaxis, np.newaxis]
        return np.ones(self.input_shape) * (delta_expanded / (height * width))
    
    def boot(self, shape: tuple):
        return
