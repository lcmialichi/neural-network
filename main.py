from neural_network.core import Neuron
from neural_network.app import App
from neural_network.board import Chart
from neural_network.initializations import He
from neural_network.activations import Relu

def main():
    model = Neuron({
        'input_size': 784,
        'hidden_size': 256,
        'output_size': 10,
        'layers_number': 4,
        'learning_rate': 0.0008,
        'regularization_lambda': 0.0001,
    }, initializer=He())

    app = App.new_instance_with_model(
        model=model, title="Reconhecimento de Dígitos"
    )
    app.model().set_activation(Relu())
    app.board().set_handler(app.predict_image)
    app.draw()
    app.train(
        data_file="data/mnist_train.csv",
        epochs=20,
        batch_size=32,
        plot=Chart().plot_activations,
    )
    app.loop()

if __name__ == "__main__":
    main()
