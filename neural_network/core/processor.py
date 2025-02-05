from abc import ABC, abstractmethod
from typing import Tuple, Generator

class Processor(ABC):
    
    @abstractmethod
    def get_train_batches(self) -> Generator[Tuple, None, None]:
        pass
    
    @abstractmethod
    def get_val_batches(self) -> Generator[Tuple, None, None]:
        pass

    @abstractmethod
    def get_test_batches(self) -> Generator[Tuple, None, None]:
        pass