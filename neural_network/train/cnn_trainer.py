from neural_network.core.image_processor import ImageProcessor
from .base_trainer import BaseTrainer
from typing import Callable, Union, Tuple
from tqdm import tqdm

class CnnTrainer(BaseTrainer):
    def train(
            self,
            epochs: int = 10, 
            plot: Union[None, Callable] = None,
    ) -> None:        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_accuracy = 0
            num_batches = 0
            
            with tqdm(self._processor.get_train_batches(), desc=f'Epoch {epoch+1}/{epochs}', unit='batch',  leave=False) as progress_bar:
                for batch_data, batch_labels in progress_bar:
                    batch_data = batch_data / 255.0
                    
                    output = self._model.train(x_batch=batch_data, y_batch=batch_labels)
                    
                    loss = self._model.get_output_loss(output, batch_labels)
                    accuracy = self._model.get_output_accuracy(output, batch_labels)
                    
                    epoch_loss += loss
                    epoch_accuracy += accuracy
                    num_batches += 1
                    
                    if plot is not None:
                        plot(epoch, num_batches, loss, accuracy)
                    
                    progress_bar.set_postfix(loss=f'{loss:.4f}', accuracy=f'{accuracy:.4f}')
            
            epoch_loss /= num_batches
            epoch_accuracy /= num_batches
            total_time = progress_bar.format_dict["elapsed"]

            print(
                f"\033[1;32mEpoch  {epoch+1}/{epochs}\033[0m"
                f" - \033[1;34mLoss\033[0m: {epoch_loss:.4f}, "
                f"\033[1;34mAccuracy\033[0m: {epoch_accuracy:.4f} "
                f"\033[1;33mbatches\033[0m: {num_batches}, "
                f"\033[1;36mtime\033[0m: {total_time:.2f} seconds"
            )