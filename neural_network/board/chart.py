import matplotlib.pyplot as plt
import numpy as np

class Chart:
    def __init__(self):
        self.__accurace = 0
        self.__total = 0

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
               
            plt.plot(predicted_index, 1, 'o', color=color, markersize=10, label="Previsão" if i == 0 else "")

        self.__total += len(true_value)
        plt.title(f'Epoch {epoch + 1} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}')
        
        plt.ylabel('Ativação')
        plt.xlabel('Índice do Neurônio (0-9)')
        plt.xticks(np.arange(10))
        plt.yticks([])

        plt.legend(loc="best")
        plt.tight_layout()

        plt.draw()
        plt.pause(0.1)
