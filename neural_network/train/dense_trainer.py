import pandas as pd
from .base_trainer import BaseTrainer
from typing import Callable, Union
import numpy as np

class DenseTrainer(BaseTrainer):
    def train(self, filename: str, epochs: int = 10, batch_size: int = 32, plot: Union[None, Callable] = None):
        data = pd.read_csv(filename)
        X = data.drop('label', axis=1).to_numpy()
        y = data['label'].values

        y_one_hot = np.zeros((y.size, self._model.get_output_size()))
        y_one_hot[np.arange(y.size), y] = 1

        total_samples = len(X)
        indices = np.random.default_rng(42).permutation(total_samples)
        x_shuffled = X[indices]
        y_shuffled = y_one_hot[indices]

        for epoch in range(epochs):
            for i in range(0, total_samples, batch_size):
                batch_x = x_shuffled[i:i + batch_size]
                batch_y = y_shuffled[i:i + batch_size]

                output = self._model.train(x_batch=batch_x, y_batch=batch_y)

                loss = self.compute_loss(output, batch_y)
                accuracy = self.compute_accuracy(output, batch_y)
                
                if plot is not None:
                    plot(output, epoch, batch_y, loss, accuracy)
