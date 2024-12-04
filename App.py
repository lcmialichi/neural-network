import tkinter as tk
from Neuron import Neuron
from DrawBoard import DrawBoard
import numpy as np
from Train import Train
from typing import Callable, Union

class App:
    def __init__(self, board: DrawBoard, model: Neuron):
        self.__model = model
        self.__board: DrawBoard = board
        self.__train: Train = Train(model)

    @staticmethod
    def new_instance_with_model(model: Neuron, title: str)-> "App":
        root = tk.Tk()
        root.title(title)

        return App(board=DrawBoard(root), model=model)
    
    def draw(self) -> None:
        self.board().draw()
        
    def model(self) -> Neuron:
        return self.__model

    def board(self) -> DrawBoard:
        return self.__board
    
    def train(self, file_name: str,  epochs: int =10, batch_size: int =32, plot: Union[None, Callable] = None ) -> None: 
        self.__train.train_with_csv(file_name, epochs, batch_size, plot)
    
    def predict_image(self, image: np.ndarray):
        return np.argmax(self.model().predict(image))
    
    def loop(self) -> None:
        self.board().root().mainloop()
    