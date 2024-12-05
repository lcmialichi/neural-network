from neural_network.core.neuron import Neuron
from neural_network.core import ImageProcessor
import pandas as pd
import numpy as np
from typing import Callable, Union
import os
from PIL import Image
import numpy as np

class Train:
    def __init__(self, model: Neuron):
        self.__model = model

    def train_with_csv(self, filename: str, epochs: int = 10, batch_size: int = 32, plot: Union[None, Callable] = None) -> None:
        data = pd.read_csv(filename)
        X = data.drop('label', axis=1).to_numpy()
        y = data['label'].values

        y_one_hot = np.zeros((y.size, self.__model.get_output_size()))
        y_one_hot[np.arange(y.size), y] = 1

        total_samples = len(X)
        indices = np.random.default_rng(42).permutation(total_samples)
        X_shuffled = X[indices]
        y_shuffled = y_one_hot[indices]

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_accuracy = 0

            for i in range(0, total_samples, batch_size):
                batch_x = X_shuffled[i:i + batch_size]
                batch_y = y_shuffled[i:i + batch_size]

                output = self.__model.train(x_batch=batch_x / 255, y_batch=batch_y)

                loss = self.compute_loss(output, batch_y)
                accuracy = self.compute_accuracy(output, batch_y)

                epoch_loss += loss
                epoch_accuracy += accuracy

                if plot is not None:
                    plot(output, epoch, batch_y, loss, accuracy)

            # Após cada epoch, mostrar o loss e accuracy médios
            print(f'Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss / total_samples}, Accuracy: {epoch_accuracy / total_samples}')

    def train_from_images(self, base_dir, image_size=(50, 50), epochs: int = 10, batch_size=32, plot: Union[None, Callable] = None):
        image_processor = ImageProcessor(base_dir, image_size, batch_size)

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_accuracy = 0

            for batch_data, batch_labels in image_processor.process_images():
                output = self.__model.train(x_batch=batch_data / 255, y_batch=batch_labels)

                loss = self.compute_loss(output, batch_labels)
                accuracy = self.compute_accuracy(output, batch_labels)

                epoch_loss += loss
                epoch_accuracy += accuracy

                if plot is not None:
                    plot(output, epoch, batch_labels, loss, accuracy)

   
    def compute_loss(self, y_pred, y_true):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))

    def compute_accuracy(self, y_pred, y_true):
        return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))
