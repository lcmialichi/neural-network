import tkinter as tk
from neural_network.core.base_network import BaseNetwork
from neural_network.board import Drawable
import numpy as np
from typing import Callable, Union

class App:
    def __init__(self, board: Drawable, model: BaseNetwork):
        self.__model = model
        self.__board: Drawable = board

    def draw(self) -> None:
        self.board().draw()
        
    def model(self) -> BaseNetwork:
        return self.__model

    def board(self) -> Drawable:
        return self.__board
    
    def train_csv(self, data_file: str,  epochs: int =10, batch_size: int =32, plot: Union[None, Callable] = None ) -> None: 
        self.__model.get_trainer().train(data_file, epochs, batch_size, plot)

    def train_images(self, base_dir, image_size=(50, 50), epochs: int = 10, batch_size=32, plot: Union[None, Callable] = None ) -> None: 
        self.__model.get_trainer().train(base_dir, image_size, epochs, batch_size, plot)
    
    def predict_image(self, image: np.ndarray):
        return np.argmax(self.model().predict(image))
    
    def loop(self) -> None:
        self.board().loop()
    