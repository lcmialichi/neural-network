import matplotlib.pyplot as plt

class Chart:
    def __init__(self):
        self.iterations = []
        self.loss_values = []
        self.accuracy_values = []

        self.cumulative_loss = 0.0
        self.cumulative_accuracy = 0.0
        self.loss_averages = []
        self.accuracy_averages = []
        self.current_iteration = 0

        plt.ion()
        plt.style.use('ggplot')
        self.fig, self.ax1 = plt.subplots(figsize=(8, 6))
        
        self.ax2 = self.ax1.twinx()

        self.loss_line, = self.ax1.plot([], [], label='Loss', color='red', linewidth=1.0)
        self.loss_avg_line, = self.ax1.plot([], [], label='Avg Loss', color='red', linestyle='--', linewidth=0.8)
        self.accuracy_line, = self.ax2.plot([], [], label='Accuracy (%)', color='green', linewidth=1.0)
        self.accuracy_avg_line, = self.ax2.plot([], [], label='Avg Accuracy (%)', color='green', linestyle='--', linewidth=0.8)

        self.ax1.set_xlabel('Iterations', fontsize=12)
        self.ax1.set_ylabel('Loss', fontsize=12, color='red')
        self.ax2.set_ylabel('Accuracy (%)', fontsize=12, color='green')
        self.ax1.grid(True, linestyle='--', alpha=0.6)

        self.fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), ncol=4, frameon=False, fontsize=10)

    def plot_metrics(self, epoch, loss, accuracy):
        self.current_iteration += 1
        self.iterations.append(self.current_iteration)
        self.loss_values.append(loss)
        self.accuracy_values.append(accuracy * 100)

        self.cumulative_loss += loss
        self.cumulative_accuracy += accuracy * 100
        avg_loss = self.cumulative_loss / len(self.iterations)
        avg_accuracy = self.cumulative_accuracy / len(self.iterations)
        self.loss_averages.append(avg_loss)
        self.accuracy_averages.append(avg_accuracy)

        self.loss_line.set_data(self.iterations, self.loss_values)
        self.loss_avg_line.set_data(self.iterations, self.loss_averages)
        self.accuracy_line.set_data(self.iterations, self.accuracy_values)
        self.accuracy_avg_line.set_data(self.iterations, self.accuracy_averages)

        self.ax1.set_xlim(0, max(self.iterations))
        self.ax1.set_ylim(0, max(max(self.loss_values), max(self.loss_averages)) + 0.5)
        self.ax2.set_ylim(0, 100)

        self.ax1.set_title(f'Training Metrics (Epoch {epoch + 1})', fontsize=14, weight='bold', pad=10)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
