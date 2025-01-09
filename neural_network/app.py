import tkinter as tk
from neural_network.core.base_network import BaseNetwork
from neural_network.board import Drawable
import numpy as np

class App:
    def __init__(self, board: Drawable, model: BaseNetwork):
        self._model = model
        self._board: Drawable = board

    def draw(self) -> None:
        self.board().draw()
        
    def model(self) -> BaseNetwork:
        return self._model

    def board(self) -> Drawable:
        return self._board
    
    def predict_image(self, image: np.ndarray):
        return np.argmax(self._model.predict(image))
    
    def loop(self) -> None:
        self.board().loop()
    