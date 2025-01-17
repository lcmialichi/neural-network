import argparse
from neural_network.console import Commands
from neural_network.app import App
from neural_network.board import Chart
from neural_network.initializations import He
from neural_network.configuration import CnnConfiguration
from neural_network.activations import LeakyRelu
from neural_network.activations import Relu
from neural_network.board import FileInput
from neural_network.core import Padding
from neural_network.activations import Softmax
from neural_network.core.image_processor import ImageProcessor
from neural_network.scheduler import ReduceLROnPlateau
from neural_network.optimizers import Adam

def create_configuration():
    config = CnnConfiguration({
        'input_shape': (3, 50, 50),
        'regularization_lambda': 0.0001,
        'optimize': True
    })

    # Processor for test, validation and training process
    config.set_processor(
        ImageProcessor(
            base_dir="./data/breast-histopathology-images",
            image_size=(50, 50),
            batch_size=64,
            split_ratios=(0.7, 0.15, 0.15),
            shuffle=True,
            rotation_range=15,
            rand_horizontal_flip=0.5,
            rand_vertical_flip=0.5,
            rand_brightness=0.2,
            rand_contrast=0.3,
            rand_crop=0.1
        )
    )
    
    # cache model state
    config.with_cache(path="./data/cache/model.pkl")
    
    config.set_global_optimizer(Adam(learning_rate=0.001))
    
    config.with_initializer(He())
    config.padding_type(Padding.SAME)
    
    # first layer
    config.add_filter(filter_number=32, filter_shape=(3, 3), activation=Relu(), stride=1)
    config.add_polling(polling_shape=(2, 2), stride=2)
    config.add_batch_normalization()
    
    # second layer
    config.add_filter(filter_number=64, filter_shape=(3, 3), activation=Relu(), stride=1)
    config.add_polling(polling_shape=(2, 2), stride=2)
    config.add_batch_normalization()
    
    # third layer
    config.add_filter(filter_number=128, filter_shape=(3, 3), activation=Relu(), stride=1)
    config.add_polling(polling_shape=(2, 2), stride=2)
    config.add_batch_normalization()

    # fourth layer
    config.add_filter(filter_number=256, filter_shape=(3, 3), activation=Relu(), stride=1)
    config.add_polling(polling_shape=(2, 2), stride=2)
    config.add_batch_normalization()
    
    # dense layers
    config.add_hidden_layer(size=512 , activation=LeakyRelu(), dropout=0.4)
    
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

def train_model(app: App, plot:bool):
    app.model().set_training_mode()
    app.model().get_trainer().train(
        epochs=10,
        plot= Chart().plot_metrics if plot else None,
        scheduler=ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6)
    )
    
def validate_model(app: App):
    app.draw()
    app.model().set_test_mode()
    app.board().set_handler(handler=app.predict_image)
    app.board().set_labels(path_json="./labels/breast_cancer.json")
    app.loop()

def test_model(app: App, plot:bool):
    app.model().set_test_mode()
    app.model().get_tester().test(plot= Chart().plot_metrics if plot else None)

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

    modes  = {
        "train": lambda: train_model(app, args.plot),
        "validate": lambda: validate_model(app),
        "test": lambda: test_model(app, args.plot),
    }

    action = modes .get(args.mode)
    action()
    
if __name__ == "__main__":
    main()
