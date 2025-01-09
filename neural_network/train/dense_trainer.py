from .base_trainer import BaseTrainer
from typing import Callable, Union

class DenseTrainer(BaseTrainer):
    def train(
            self, 
            base_dir: str = "",
            image_size=(50, 50), 
            epochs: int = 10, 
            batch_size=32, 
            plot: Union[None, Callable] = None,
            rotation_range: int = 30
    ):
      ...
