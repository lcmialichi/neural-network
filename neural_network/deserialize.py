import pickle
import os
from neural_network.gcpu import driver

class Deserialize(pickle.Unpickler):
     def find_class(self, module, name): 
        if module == 'cupy._core.core'  and name == 'array':
            return driver.gcpu.array
        return super().find_class(module, name)