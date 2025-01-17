from neural_network.core.image_processor import ImageProcessor
from neural_network.core.scheduler import Scheduler
from .base_trainer import BaseTrainer
from typing import Callable, Union, Tuple
from tqdm import tqdm

class CnnTrainer(BaseTrainer):
    def train(
            self,
            epochs: int = 10, 
            plot: Union[None, Callable] = None,
            scheduler: Union[None, Scheduler] = None
    ) -> None:
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_accuracy = 0
            num_batches = 0 
            self._model.set_training_mode()
            with tqdm(self._processor.get_train_batches(), desc=f'Epoch {epoch+1}/{epochs} (run)', unit='batch',  leave=False) as progress_bar:
                for batch_data, batch_labels in progress_bar:
                    batch_data = batch_data / 255.0
                    
                    output = self._model.train(x_batch=batch_data, y_batch=batch_labels)
                    
                    loss = self._model.get_output_loss(output, batch_labels)
                    accuracy = self._model.get_output_accuracy(output, batch_labels)
                    
                    epoch_loss += loss
                    epoch_accuracy += accuracy
                    num_batches += 1
                    
                    if plot is not None:
                        plot(epoch, loss, accuracy)
                    
                    progress_bar.set_postfix(loss=f'{loss:.4f}', accuracy=f'{accuracy:.4f}')
            
            epoch_loss /= num_batches
            epoch_accuracy /= num_batches
            total_time = progress_bar.format_dict["elapsed"]

            print(
                f"\033[1;32mEpoch {epoch+1}/{epochs} (avg)\033[0m"
                f" - \033[1;34mLoss\033[0m: {epoch_loss:.4f}, "
                f"\033[1;34mAccuracy\033[0m: {epoch_accuracy:.4f} "
                f"\033[1;33mbatches\033[0m: {num_batches}, "
                f"\033[1;36mtime\033[0m: {total_time:.2f} seconds",
            )

            loss, accurracy = self._validate(epoch)
            print(
                f"\033[1;36mEpoch {epoch+1}/{epochs} (val)\033[0m"
                f" - \033[1;34mLoss\033[0m: {loss:.4f}, "
                f"\033[1;34mAccuracy\033[0m: {accurracy:.4f}"
            )
            
            if scheduler:
                scheduler(self._model, loss, accurracy)

    def _validate(self, epoch: int) -> Tuple[float, float]:
        val_loss = 0
        val_accuracy = 0
        num_batches = 0
        self._model.set_test_mode()
        with tqdm(self._processor.get_val_batches(), desc=f'Epoch {epoch+1} (val)', unit='batch', leave=False) as progress_bar:
            for batch_data, batch_labels in progress_bar:
                batch_data = batch_data / 255.0

                output = self._model.predict(batch_data)

                loss = self._model.get_output_loss(output, batch_labels)
                accuracy = self._model.get_output_accuracy(output, batch_labels)

                val_loss += loss
                val_accuracy += accuracy
                num_batches += 1


                progress_bar.set_postfix(loss=f'{loss:.4f}', accuracy=f'{accuracy:.4f}')

        val_loss /= num_batches
        val_accuracy /= num_batches

        return val_loss, val_accuracy