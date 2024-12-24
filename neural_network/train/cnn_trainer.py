from neural_network.core.image_processor import ImageProcessor
from .base_trainer import BaseTrainer
from typing import Callable, Union

class CnnTrainer(BaseTrainer):
    def train(self, base_dir: str = "", image_size=(50, 50), epochs: int = 10, batch_size=32, plot: Union[None, Callable] = None):
        image_processor = ImageProcessor(base_dir, image_size, batch_size)
        for epoch in range(epochs):
            for batch_data, batch_labels in image_processor.process_images():
                output = self._model.train(x_batch=(batch_data / 255 - 0.5) / 0.5, y_batch=batch_labels)
                loss = self.compute_loss(output, batch_labels)
                accuracy = self.compute_accuracy(output, batch_labels)
                print(f"\rEpoch: {epoch+1}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}", end="", flush=True)
                if plot is not None:
                    plot(output, epoch, batch_labels, loss, accuracy)
