from tqdm import tqdm
from typing import Union, Callable
from neural_network.core.processor import Processor

class Tester:
    def __init__(self, model):
        self._model = model

    def test(self, processor: Processor, plot: Union[None, Callable] = None):
        num_batches = 0
        test_loss, avg_loss = 0, 0
        test_accuracy, avg_accuracy = 0, 0

        with tqdm(processor.get_test_batches(), desc='Testing', unit='batch', leave=False) as progress_bar:
             for batch_data, batch_labels in progress_bar:
                batch_data = batch_data / 255.0

                output = self._model.predict(batch_data)

                test_loss += self._model.get_output_loss(output, batch_labels)
                test_accuracy += self._model.get_output_accuracy(output, batch_labels)
                num_batches += 1
                
                avg_loss = test_loss / num_batches
                avg_accuracy = test_accuracy / num_batches

                if plot is not None:
                    plot(0, avg_loss, avg_accuracy, 'Test Metrics')
                    
                progress_bar.set_postfix(loss=f'{avg_loss:.4f}', accuracy=f'{avg_accuracy:.4f}')

        total_time = progress_bar.format_dict["elapsed"]

        print(
            f" - \033[1;34mLoss\033[0m: {avg_loss:.4f}, "
            f"\033[1;34mAccuracy\033[0m: {avg_accuracy:.4f} "
            f"\033[1;33mbatches\033[0m: {num_batches}, "
            f"\033[1;36mtime\033[0m: {total_time:.2f} seconds"
        )