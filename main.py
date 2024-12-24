from neural_network.app import App
from neural_network.board import Chart
from neural_network.initializations import He
from neural_network.configuration import CnnConfiguration
from neural_network.activations import LeakyRelu
from neural_network.board import FileInput
from neural_network.core import Padding

def main():
    
    config = CnnConfiguration({
        'input_shape': (3, 50, 50),
        'output_size': 2,
        'learning_rate': 0.0001,
        'regularization_lambda': 0.0001,
        'dropout_rate': 0.3,
        'optimize': True
    })
    
    config.with_initializer(He())
    config.padding_type(Padding.SAME)
    
    config.add_hidden_layer(size=128, activation=LeakyRelu())
    config.add_hidden_layer(size=256, activation=LeakyRelu())              
    
    config.add_filter(filter_number=8, filter_shape=(3, 3), activation=LeakyRelu(), stride=1)
    config.add_filter(filter_number=16, filter_shape=(3, 3), activation=LeakyRelu(), stride=1)
    config.add_filter(filter_number=16, filter_shape=(3, 3), activation=LeakyRelu(), stride=1)
    config.add_filter(filter_number=32, filter_shape=(3, 3), activation=LeakyRelu(), stride=1)
    
    app = App(
        model=config.new_model(), 
        board=FileInput(
            title="Reconhecimento de cancer de mama",
            img_resize=(50,50),
            )
        )
    
    app.draw()
    app.board().set_handler(handler=app.predict_image)
    app.board().set_labels(path_json="./labels/breast_cancer.json")

    app.train_images(
        base_dir="./data/breast-histopathology-images",
        image_size=(50,50),
        epochs=10,
        batch_size=64,
        # plot=Chart(size=2).plot_activations
    )
    
    app.loop()

if __name__ == "__main__":
    main()