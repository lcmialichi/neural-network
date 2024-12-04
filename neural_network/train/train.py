from neural_network.core.neuron import Neuron
import pandas as pd
import numpy as np
from typing import Callable, Union

class Train:
    def __init__(self, model: Neuron):
        self.__model = model

    def train_with_csv(self, filename: str, epochs: int = 10, batch_size: int = 32, plot: Union[None, Callable] = None) -> None:
        data = pd.read_csv(filename)
        X = data.drop('label', axis=1).to_numpy()
        y = data['label'].values

        y_one_hot = np.zeros((y.size, 10))
        y_one_hot[np.arange(y.size), y] = 1

        total_samples = len(X)

        for epoch in range(epochs):
            indices = np.random.default_rng(42).permutation(total_samples)
            x_shuffled = X[indices]
            y_shuffled = y_one_hot[indices]

            for i in range(0, total_samples, batch_size):
                batch_x = x_shuffled[i:i + batch_size]
                batch_y = y_shuffled[i:i + batch_size]

                output = self.__model.train(x_batch=batch_x / 255, y_batch=batch_y)

                if plot is not None:
                    loss = self.compute_loss(output, batch_y)
                    accuracy = self.compute_accuracy(output, batch_y)
                    plot(output, epoch, batch_y, loss, accuracy)

    def compute_loss(self, y_pred, y_true):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))

    def compute_accuracy(self, y_pred, y_true):
        return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))
