from neural_network.core import CnnNetwork
from neural_network.app import App
from neural_network.board import Chart
from neural_network.initializations import He
from neural_network.activations import Relu
from neural_network.board import FileInput

def main():
    model = CnnNetwork({
        'input_size': 50 * 50 * 3,
        'hidden_size': 512,
        'output_size': 2,
        'layers_number': 4,
        'learning_rate': 0.0001,
        'regularization_lambda': 0.0001,
        'dropout_rate': 0.3
    })

    app = App.new_instance_with_model(
        model=model, board=FileInput(
            title="Reconhecimento de cancer de mama",
            img_resize=(50,50),
            )
        )
    
    app.draw()
    app.model().set_activation(Relu())
    app.board().set_handler(handler=app.predict_image)
    app.board().set_labels(path_json="./labels/breast_cancer.json")

    app.train_images(
        base_dir="./data/breast-histopathology-images",
        image_size=(50,50),
        epochs=10,
        batch_size=32,
        plot=Chart(2).plot_activations
    )
    
    app.loop()

if __name__ == "__main__":
    main()