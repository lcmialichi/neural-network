import argparse
from neural_network.console import Commands
from neural_network.app import App
from neural_network.board import Chart
from neural_network.initializations import He
from neural_network.configuration import CnnConfiguration
from neural_network.activations import LeakyRelu
from neural_network.board import FileInput
from neural_network.core import Padding
from neural_network.activations import Softmax

def create_configuration():
    config = CnnConfiguration({
        'input_shape': (3, 50, 50),
        'learning_rate': 0.001,
        'regularization_lambda': 0.0001,
        'dropout': 0.1,
        'optimize': True
    })
    
    config.with_initializer(He(path="./data/cache/he.pkl"))
    config.padding_type(Padding.SAME)
    
    # first layer
    config.add_filter(filter_number=16, filter_shape=(3, 3), activation=LeakyRelu(), stride=1)
    config.add_polling(polling_shape=(2, 2), stride=2)
    
    # second layer
    config.add_filter(filter_number=32, filter_shape=(3, 3), activation=LeakyRelu(), stride=1)
    
    # third layer
    config.add_filter(filter_number=64, filter_shape=(3, 3), activation=LeakyRelu(), stride=1)
    config.add_polling(polling_shape=(2, 2), stride=2)
    config.add_batch_normalization()
    
    # dense layers
    config.add_hidden_layer(size=128, activation=LeakyRelu())
    config.add_hidden_layer(size=256, activation=LeakyRelu())
    config.add_hidden_layer(size=512, activation=LeakyRelu())
    
    # output
    config.output(size=2, activation=Softmax())
    
    return config

def create_app(config: CnnConfiguration) -> App: 
    return App(
        model=config.new_model(), 
        board=FileInput(
            title="Breast cancer recognition",
            img_resize=(50, 50),
        )
    )

def train_model(app: App):
    app.model().set_training_mode()
    app.train_images(
        base_dir="./data/breast-histopathology-images",
        image_size=(50, 50),
        epochs=10,
        batch_size=32
    )
    
def test_model(app: App):
    app.draw()
    app.model().set_test_mode()
    app.board().set_handler(handler=app.predict_image)
    app.board().set_labels(path_json="./labels/breast_cancer.json")
    app.loop()

def main():
    commands = Commands(argparse.ArgumentParser(description="train or test the model"))
    commands.register()
    
    args = commands.get_args()
    config = create_configuration()
    
    if args.clear_cache:
        config.restore_initialization_cache()
    
    if args.no_cache:
        config.with_no_cache()
    
    app = create_app(config)
    
    if args.mode == "train":
        train_model(app)
    elif args.mode == "test":
        test_model(app)
    

if __name__ == "__main__":
    main()
