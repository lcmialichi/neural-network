from neural_network.core import CnnNetwork
from neural_network.app import App
from neural_network.board import Chart
from neural_network.initializations import He
from neural_network.activations import Relu
from neural_network.board import FileInput
from neural_network.core import Padding

def main():
    
    config = CnnNetwork.config({
        'input_shape': (3, 50, 50),
        'output_size': 2,
        'learning_rate': 0.0001,
        'regularization_lambda': 0.0001,
        'dropout_rate': 0.3,
        'stride': 1,
        'optimize': True
    })    
            
    config.add_hidden_layer(size=128)                   
    config.add_hidden_layer(size=128)                   
    config.add_filter(filter_number=16, filter_shape=(2, 2))         
    config.add_filter(filter_number=32, filter_shape=(3, 3))                 
    config.with_activation(Relu())                      
    config.with_initializer(He())                       
            
    model = CnnNetwork(config)

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