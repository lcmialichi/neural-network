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
            callbacks: list = None
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
                    
                    epoch_loss += self._model.get_output_loss(output, batch_labels)
                    epoch_accuracy += self._model.get_output_accuracy(output, batch_labels)
                    num_batches += 1
                    
                    avg_loss = epoch_loss / num_batches
                    avg_accuracy = epoch_accuracy / num_batches

                    if plot is not None:
                        plot(epoch, avg_loss, avg_accuracy)
                    
                    self._model.step()
                    progress_bar.set_postfix(loss=f'{avg_loss:.4f}', accuracy=f'{avg_accuracy:.4f}')
            
            total_time = progress_bar.format_dict["elapsed"]
            self._model.save_state()
            
            print(
                f"\033[1;32mEpoch {epoch+1}/{epochs} (avg)\033[0m"
                f" - \033[1;34mLoss\033[0m: {avg_loss:.4f}, "
                f"\033[1;34mAccuracy\033[0m: {avg_accuracy:.4f} "
                f"\033[1;33mbatches\033[0m: {num_batches}, ",
                f"\033[1;32mLearning rate\033[0m: {self._model.get_learning_rate()}, "
                f"\033[1;36mtime\033[0m: {total_time:.2f} seconds",
            )

            val_loss, val_accuracy = self._validate(epoch)

            print(
                f"\033[1;36mEpoch {epoch+1}/{epochs} (val)\033[0m"
                f" - \033[1;34mLoss\033[0m: {val_loss:.4f}, "
                f"\033[1;34mAccuracy\033[0m: {val_accuracy:.4f}"
            )
            metrics = {
                'val_loss': val_loss,
                'val_accurracy': val_accuracy,
                'loss': avg_loss,
                'accurracy': avg_accuracy
            }

            self._add_history(metrics)
            for callback in callbacks:
                callback(self._model, metrics)

    def _validate(self, epoch: int) -> Tuple[float, float]:
        loss = 0
        accuracy = 0
        num_batches = 0
        self._model.set_test_mode()
        with tqdm(self._processor.get_val_batches(), desc=f'Epoch {epoch+1} (val)', unit='batch', leave=False) as progress_bar:
            for batch_data, batch_labels in progress_bar:
                
                output = self._model.predict(batch_data / 255.0)

                loss += self._model.get_output_loss(output, batch_labels)
                accuracy += self._model.get_output_accuracy(output, batch_labels)
                    
                num_batches += 1
                avg_loss = loss / num_batches
                avg_accuracy = accuracy / num_batches

                progress_bar.set_postfix(loss=f'{avg_loss:.4f}', accuracy=f'{avg_accuracy:.4f}')
                
        return avg_loss, avg_accuracy