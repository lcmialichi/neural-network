import argparse
from neural_network.console import Commands
from neural_network.app import App
from neural_network.board import Chart
from neural_network.initializations import He
from neural_network.configuration import CnnConfiguration
from neural_network.activations import LeakyRelu, Relu, Softmax, Sigmoid
from neural_network.board import FileInput
from neural_network.core import Padding
from neural_network.core.image_processor import ImageProcessor
from neural_network.scheduler import ReduceLROnPlateau
from neural_network.optimizers import Adam
from neural_network.foundation import Kernel, Output, HiddenLayer
from neural_network.loss import CrossEntropyLoss, BinaryCrossEntropyLoss

def create_configuration():
    config = CnnConfiguration({
        'input_shape': (3, 100, 100),
        'regularization_lambda': 0.0001,
    })

    # Processor for test, validation, and training
    config.set_processor(
        ImageProcessor(
            base_dir="./data/breast-histopathology-images",
            image_size=(100, 100),
            batch_size=8,
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

    # Cache model state
    config.with_cache(path="./data/cache/model_optimized.pkl")
    config.set_global_optimizer(Adam(learning_rate=0.001))
    config.padding_type(Padding.SAME)

    # Convolutional blocks
    # Block 1
    kernel: Kernel = config.add_kernel(number=8, shape=(7, 7), stride=1)
    kernel.initializer(He())
    kernel.activation(LeakyRelu())
    kernel.max_pooling(shape=(2, 2), stride=2)
    
    kernel: Kernel = config.add_kernel(number=8, shape=(3, 3), stride=1)
    kernel.initializer(He())
    kernel.activation(Relu())
    kernel.batch_normalization()

    # Fully connected layers
    layer: HiddenLayer = config.add_hidden_layer(size=256, dropout=0.5)
    layer.activation(LeakyRelu())
    layer.initializer(He())

    # Output layer
    output: Output = config.output(size=1)  
    output.activation(Sigmoid())
    output.initializer(He())
    output.loss_function(BinaryCrossEntropyLoss())

    return config

def create_app(config: CnnConfiguration) -> App: 
    return App(
        model=config.new_model(), 
        board=FileInput(
            title="Breast Cancer Recognition",
            img_resize=(100, 100),
        )
    )

def train_model(app: App, plot: bool):
    app.model().set_training_mode()
    app.model().get_trainer().train(
        epochs=50,
        plot=Chart().plot_metrics if plot else None,
        scheduler=ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
    )

def validate_model(app: App):
    app.draw()
    app.model().set_test_mode()
    app.board().set_handler(handler=app.predict_image)
    app.board().set_labels(path_json="./labels/breast_cancer.json")
    app.loop()

def test_model(app: App, plot: bool):
    app.model().set_test_mode()
    app.model().get_tester().test(plot=Chart().plot_metrics if plot else None)

def main():
    commands = Commands(argparse.ArgumentParser(description="Train or test the model"))
    commands.register()

    args = commands.get_args()
    config = create_configuration()

    if args.clear_cache: 
        config.restore_initialization_cache()

    if args.no_cache: 
        config.with_no_cache() 

    app = create_app(config)

    modes = {
        "train": lambda: train_model(app, args.plot),
        "validate": lambda: validate_model(app),
        "test": lambda: test_model(app, args.plot),
    }

    action = modes.get(args.mode)
    action()

if __name__ == "__main__":
    main()
