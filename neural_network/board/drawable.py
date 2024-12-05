from abc import ABC, abstractmethod
from typing import Callable, Union
from neural_network.core import LabelLoader

class Drawable(ABC):
    @property
    def handler(self) -> Union[None, Callable]:
        return getattr(self, '_handler', None)

    @handler.setter
    def handler(self, value: Callable) -> None:
        self._handler = value

    @property
    def labels(self) -> Union[None, LabelLoader]:
        return getattr(self, '_labels', None)

    @labels.setter
    def labels(self, value: LabelLoader) -> None:
        self._labels = value
    
    @abstractmethod
    def draw(self) -> None:
        pass
    
    @abstractmethod
    def loop(self):
        pass

    def set_handler(self, handler: Callable) -> None:
        self.handler = handler
        
    def get_handler(self) -> Union[None, Callable]:
        return self.handler
    
    def set_labels(self, path_json: str) -> None:
        self.labels = LabelLoader(path_json)

    def get_label(self, index: int):
        if self.labels is None:
            return index
        
        return self.labels.get_label(index)

  

    