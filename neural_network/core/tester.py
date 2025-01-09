from tqdm import tqdm

class Tester:
    def __init__(self, model, processor):
        self._model = model
        self._processor = processor

    def test(self):
        num_batches = 0
        test_loss = 0
        test_accuracy = 0

        with tqdm(self._processor.get_test_batches(), desc='Testing', unit='batch') as progress_bar:
             for batch_data, batch_labels in progress_bar:
                batch_data = batch_data / 255.0

                output = self._model.predict(batch_data)

                loss = self._model.get_output_loss(output, batch_labels)
                accuracy = self._model.get_output_accuracy(output, batch_labels)
                
                test_loss += loss
                test_accuracy += accuracy
                num_batches += 1

                progress_bar.set_postfix(loss=f'{loss:.4f}', accuracy=f'{accuracy:.4f}')

        test_loss /= num_batches
        test_accuracy /= num_batches
            
        print(f"General - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")