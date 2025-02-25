import pickle
import os
from neural_network.deserialize import Deserialize

class Storage():
    def __init__(self, path: str):
        self._path = path
        
    def store(self, model):
        directory = os.path.dirname(self._path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        try:
            with open(self._path, 'wb') as f:
                pickle.dump(model, f)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            pass
    
    def get(self):
        try:
            with open(self._path, 'rb') as f:
                deserializer = Deserialize(f)
                return deserializer.load()
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            raise SystemError(f"Unable to load model from cached file: '{self._path}'")
        
    def remove(self):
        if self._path is not None:
            os.remove(self._path)
            
    def has(self):
        return  os.path.exists(self._path)
    
   