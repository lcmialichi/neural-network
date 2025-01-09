from abc import ABC, abstractmethod
from typing import Tuple, Generator
import numpy as np

class Processor(ABC):
    
    @abstractmethod
    def get_train_batches(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        pass
    
    @abstractmethod
    def get_val_batches(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        pass

    @abstractmethod
    def get_test_batches(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        pass