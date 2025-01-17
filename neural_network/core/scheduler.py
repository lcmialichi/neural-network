from abc import ABC, abstractmethod

class Scheduler(ABC):
    
    @abstractmethod
    def __call__(self, model, val_loss, val_accuracy):
        pass