import matplotlib.pyplot as plt
import numpy as np

class Chart:
    def __init__(self, size: int = 10):
        self.__accurace = 0
        self.__size = size

    def plot_activations(self, activation, epoch, true_value, loss, accuracy):
        plt.clf()
        plt.subplot(1, 1, 1)
        for i in range(len(true_value)):
            predicted_index = np.argmax(activation[i])
            true_index = np.argmax(true_value[i])
            color = 'red'
            if predicted_index == true_index:
                color = 'green'
                self.__accurace += 1
               
            plt.plot(predicted_index, i, 'o', color=color, markersize=10, label="Previsão" if i == 0 else "")

        plt.title(f'Epoch {epoch + 1} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}')
        
        plt.ylabel('Ativação')
        plt.xlabel(f'Índice do Neurônio (0-{self.__size})')
        plt.xticks(np.arange(self.__size))
        plt.yticks([])

        plt.legend(loc="best")
        plt.tight_layout()

        plt.draw()
        plt.pause(0.1)
