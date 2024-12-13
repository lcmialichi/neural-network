from neural_network.core import DenseNeuron
from neural_network.app import App
from neural_network.board import DrawBoard
from neural_network.board import Chart
from neural_network.initializations import He
from neural_network.activations import Relu

def main():
    model = DenseNeuron({
        'input_size': 784,
        'hidden_size': 256,
        'output_size': 10,
        'layers_number': 4,
        'learning_rate': 0.0008,
        'regularization_lambda': 0.0001,
    }, initializer=He())

    app = App.new_instance_with_model(
        model=model, board=DrawBoard(
            title="Reconhecimento de Dígitos",
            size=280,
            line_weight=7
        )
    )
    app.model().set_activation(Relu())
    app.board().set_handler(handler=app.predict_image)
    app.board().set_labels(path_json="./labels/number.json")
    app.draw()
    app.train_csv(
        data_file="data/numbers/mnist_train.csv",
        epochs=20,
        batch_size=32,
        plot=Chart().plot_activations,
    )

    app.loop()

if __name__ == "__main__":
    main()
