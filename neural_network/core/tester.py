from tqdm import tqdm
from typing import Union, Callable

class Tester:
    def __init__(self, model, processor):
        self._model = model
        self._processor = processor

    def test(self, plot: Union[None, Callable] = None):
        num_batches = 0
        test_loss = 0
        test_accuracy = 0

        with tqdm(self._processor.get_test_batches(), desc='Testing', unit='batch', leave=False) as progress_bar:
             for batch_data, batch_labels in progress_bar:
                batch_data = batch_data / 255.0

                output = self._model.predict(batch_data)

                loss = self._model.get_output_loss(output, batch_labels)
                accuracy = self._model.get_output_accuracy(output, batch_labels)
                
                test_loss += loss
                test_accuracy += accuracy
                num_batches += 1

                if plot is not None:
                    plot(1, num_batches, loss, accuracy)
                    
                progress_bar.set_postfix(loss=f'{loss:.4f}', accuracy=f'{accuracy:.4f}')

        test_loss /= num_batches
        test_accuracy /= num_batches
        total_time = progress_bar.format_dict["elapsed"]

        print(
            f" - \033[1;34mLoss\033[0m: {test_loss:.4f}, "
            f"\033[1;34mAccuracy\033[0m: {test_accuracy:.4f} "
            f"\033[1;33mbatches\033[0m: {num_batches}, "
            f"\033[1;36mtime\033[0m: {total_time:.2f} seconds"
        )