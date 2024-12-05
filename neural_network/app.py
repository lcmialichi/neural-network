import tkinter as tk
from neural_network.core import Neuron
from neural_network.board import Drawable
import numpy as np
from neural_network.train import Train
from typing import Callable, Union
from neural_network.core import LabelLoader

class App:
    def __init__(self, board: Drawable, model: Neuron):
        self.__model = model
        self.__board: Drawable = board
        self.__train: Train = Train(model)

    @staticmethod
    def new_instance_with_model(model: Neuron, board: Drawable)-> "App":
        return App(board=board, model=model)
    
    def draw(self) -> None:
        self.board().draw()
        
    def model(self) -> Neuron:
        return self.__model

    def board(self) -> Drawable:
        return self.__board
    
    def train_csv(self, data_file: str,  epochs: int =10, batch_size: int =32, plot: Union[None, Callable] = None ) -> None: 
        self.__train.train_with_csv(data_file, epochs, batch_size, plot)

    def train_images(self, base_dir, image_size=(50, 50), epochs: int = 10, batch_size=32, plot: Union[None, Callable] = None ) -> None: 
        self.__train.train_from_images(base_dir, image_size, epochs, epochs, plot)
    
    def predict_image(self, image: np.ndarray):
        return np.argmax(self.model().predict(image))
    
    def loop(self) -> None:
        self.board().loop()
    