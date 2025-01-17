from abc import ABC, abstractmethod
from typing import Tuple, Generator
from neural_network.gcpu import gcpu

class Processor(ABC):
    
    @abstractmethod
    def get_train_batches(self) -> Generator[Tuple[gcpu.ndarray, gcpu.ndarray], None, None]:
        pass
    
    @abstractmethod
    def get_val_batches(self) -> Generator[Tuple[gcpu.ndarray, gcpu.ndarray], None, None]:
        pass

    @abstractmethod
    def get_test_batches(self) -> Generator[Tuple[gcpu.ndarray, gcpu.ndarray], None, None]:
        pass