import argparse
from neural_network.console import Commands
from neural_network.app import App
from neural_network.board import Chart
from neural_network.initializations import He, Xavier
from neural_network.configuration import CnnConfiguration
from neural_network.activations import Relu, Softmax
from neural_network.board import FileInput
from neural_network.core.padding import Padding
from neural_network.core.image_processor import ImageProcessor
from neural_network.scheduler import ReduceLROnPlateau
from neural_network.optimizers import Adam
from neural_network.foundation import Kernel, Output, HiddenLayer
from neural_network.loss import CrossEntropyLoss

IMAGE_SIZE = (50, 50)
IMAGE_CHANNELS = 3
BATCH_SIZE = 64
EPOCHS = 30

def create_configuration():
    config = CnnConfiguration({
        'input_shape': (IMAGE_CHANNELS, *IMAGE_SIZE),
        'regularization_lambda': 1e-4,
    })

    config.set_processor(
        ImageProcessor(
            base_dir="./data/breast-histopathology-images",
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            split_ratios=(0.7, 0.15, 0.15),
            shuffle=True,
            rotation_range=30,
            rand_horizontal_flip=0.7,
            rand_vertical_flip=0.7,
            rand_brightness=0.3,
            rand_contrast=0.4,
            rand_crop=0.2,
        )
    )

    config.set_global_optimizer(Adam(learning_rate=0.001))
    config.with_cache(path='./data/cache/model.pkl')
    config.padding_type(Padding.SAME)
    config.loss_function(CrossEntropyLoss())

    kernel = config.add_kernel(number=8, shape=(5, 5), stride=1)
    kernel.initializer(He())
    kernel.activation(Relu())
    kernel.batch_normalization()

    kernel1 = config.add_kernel(number=8, shape=(3, 3), stride=1)
    kernel1.initializer(He())
    kernel1.activation(Relu())
    kernel1.batch_normalization()
    kernel1.max_pooling(shape=(2, 2), stride=2)

    kernel2 = config.add_kernel(number=16, shape=(3, 3), stride=1)
    kernel2.initializer(He())
    kernel2.activation(Relu())
    kernel2.batch_normalization()

    kernel3 = config.add_kernel(number=32, shape=(3, 3), stride=1)
    kernel3.initializer(He())
    kernel3.activation(Relu())
    kernel3.batch_normalization()
    kernel3.max_pooling(shape=(2, 2), stride=2)

    config.flatten()
    dense = config.dense()
    
    # 🔹 Fully Connected
    layer1 = dense.add_layer(size=512, dropout=0.5)
    layer1.initializer(He())
    layer1.activation(Relu())

    layer2 = dense.add_layer(size=256, dropout=0.5)
    layer2.initializer(He())
    layer2.activation(Relu())

    # 🔹 Saída (Softmax para classificação)
    output = dense.add_layer(size=2)
    output.activation(Softmax())
    output.initializer(Xavier())

    return config

def create_app(config: CnnConfiguration) -> App: 
    return App(
        model=config.new_model(), 
        board=FileInput(
            title="Breast Cancer Recognition",
            img_resize=IMAGE_SIZE,
        )
    )

def train_model(app: App, plot: bool):
    app.model().set_training_mode()
    app.model().get_trainer().train(
        epochs=EPOCHS,
        plot=Chart().plot_metrics if plot else None,
        scheduler=ReduceLROnPlateau(factor=0.3, patience=5, min_lr=1e-6)
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
