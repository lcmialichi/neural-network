from neural_network.core.image_processor import ImageProcessor
from neural_network.core.scheduler import Scheduler
from .base_trainer import BaseTrainer
from typing import Callable, Union, Tuple
from tqdm import tqdm
from neural_network.core.processor import Processor

class CnnTrainer(BaseTrainer):
    def train(
            self,
            processor: Processor,
            epochs: int = 10, 
            plot: Union[None, Callable] = None,
            callbacks: list = None
    ) -> None:
        for epoch in range(epochs):
            epoch_loss, avg_loss = 0, 0
            epoch_accuracy, avg_accuracy = 0, 0
            num_batches = 0 
            self._model.set_training_mode()

            print(f"\033[1;32mEpoch {epoch+1}/{epochs}: \033[0m")

            total_sampple = len(processor.train_sample) / processor.batch_size
            current_batch = 0
            with tqdm(processor.get_train_batches(), 
                    desc=f'Batch {current_batch}/{current_batch}',
                    total=total_sampple,
                    dynamic_ncols=True, 
                    unit='batch',  
                    leave=False) as progress_bar:
                for batch_data, batch_labels in progress_bar:
                    current_batch += 1
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
            
            self._model.save_state()
            val_loss, val_accuracy = self._validate(epoch, processor)

            print(
                f"\033[1;32mEpoch {epoch+1}: \033[0m"
                f" \033[1;34mloss\033[0m: {avg_loss:.4f}, "
                f"\033[1;34maccuracy\033[0m: {avg_accuracy:.4f} "
                f"\033[1;33mbatches\033[0m: {num_batches}, ",
                f" - \033[1;34mval_loss\033[0m: {val_loss:.4f}, "
                f"\033[1;34mval_accuracy\033[0m: {val_accuracy:.4f}"
                f"\033[1;32mlearning rate\033[0m: {self._model.get_learning_rate()}"
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

    def _validate(self, epoch: int, processor: Processor) -> Tuple[float, float]:
        loss, avg_loss = 0, 0
        accuracy, avg_accuracy = 0, 0
        num_batches = 0
        self._model.set_test_mode()
        with tqdm(processor.get_val_batches(), desc=f'Epoch {epoch+1} (val)', unit='batch', leave=False) as progress_bar:
            for batch_data, batch_labels in progress_bar:
                
                output = self._model.predict(batch_data / 255.0)

                loss += self._model.get_output_loss(output, batch_labels)
                accuracy += self._model.get_output_accuracy(output, batch_labels)
                    
                num_batches += 1
                avg_loss = loss / num_batches
                avg_accuracy = accuracy / num_batches

                progress_bar.set_postfix(loss=f'{avg_loss:.4f}', accuracy=f'{avg_accuracy:.4f}')
                
        return avg_loss, avg_accuracy