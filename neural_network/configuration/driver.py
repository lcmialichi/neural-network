from enum import Enum

class Driver(Enum):
    gpu = {'module': 'cupy'}
    cpu = {'module': 'numpy'}

    def get_module(self) -> str:
        return self.value.get('module', None)
