import matplotlib.pyplot as plt

class Chart:
    def __init__(self):
        self.iterations = []
        self.loss_values = []
        self.accuracy_values = []

        plt.ion()
        plt.style.use('ggplot')
        self.fig, self.ax1 = plt.subplots(figsize=(8, 6))
        
        self.ax2 = self.ax1.twinx()

        self.loss_line, = self.ax1.plot([], [], label='Loss', color='red', linewidth=1.5)
        self.accuracy_line, = self.ax2.plot([], [], label='Accuracy (%)', color='green', linewidth=1.5)

        self.ax1.set_xlabel('Iterations', fontsize=12)
        self.ax1.set_ylabel('Loss', fontsize=12, color='red')
        self.ax2.set_ylabel('Accuracy (%)', fontsize=12, color='green')
        self.ax1.grid(True, linestyle='--', alpha=0.6)

        self.fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=2, frameon=False, fontsize=10)

    def plot_metrics(self, epoch, iteration, loss, accuracy):
        self.iterations.append(iteration)
        self.loss_values.append(loss)
        self.accuracy_values.append(accuracy * 100)

        self.loss_line.set_data(self.iterations, self.loss_values)
        self.accuracy_line.set_data(self.iterations, self.accuracy_values)

        self.ax1.set_xlim(0, max(self.iterations))
        self.ax1.set_ylim(0, max(self.loss_values) + 0.5)
        self.ax2.set_ylim(0, 100)

        self.ax1.set_title(f'Training Metrics (Epoch {epoch + 1})', fontsize=14, weight='bold', pad=20)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
