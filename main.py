import argparse
import neural_network as nn
import neural_network.supply as attr
from neural_network import Config
from neural_network.board import Chart, FileInput
from neural_network.core.padding import Padding
from neural_network.core.image_processor import ImageProcessor
from neural_network.console import Commands

IMAGE_SIZE = (50, 50)
IMAGE_CHANNELS = 3
BATCH_SIZE = 75
EPOCHS = 25

def create_configuration():
    config = Config()

    # using GPU, if not availible switch to 'cpu'
    config.driver('cpu')

    # cache model to this file
    config.with_cache(path='./data/cache/model.pkl')

    config.set_global_optimizer(attr.Adam(learning_rate=0.001))
    config.padding_type(Padding.SAME)
    config.loss_function(attr.BinaryCrossEntropyLoss())

    # ---- Convolutional Layers ----'
    kernel = config.add_kernel(number=32, shape=(3, 3), stride=1)
    kernel.initializer(attr.HeUniform())
    kernel.activation(attr.Relu())
    kernel.batch_normalization()

    kernel = config.add_kernel(number=32, shape=(3, 3), stride=1)
    kernel.initializer(attr.HeUniform())
    kernel.activation(attr.Relu())
    kernel.batch_normalization()
    kernel.max_pooling(shape=(2, 2), stride=2)

    kernel = config.add_kernel(number=64, shape=(3, 3), stride=1)
    kernel.initializer(attr.HeUniform())
    kernel.activation(attr.Relu())
    kernel.batch_normalization()

    kernel = config.add_kernel(number=64, shape=(3, 3), stride=1)
    kernel.initializer(attr.HeUniform())
    kernel.activation(attr.Relu())
    kernel.batch_normalization()
    kernel.max_pooling(shape=(2, 2), stride=2)

    kernel = config.add_kernel(number=128, shape=(3, 3), stride=1)
    kernel.initializer(attr.HeUniform())
    kernel.activation(attr.Relu())
    kernel.batch_normalization()

    kernel = config.add_kernel(number=128, shape=(3, 3), stride=1)
    kernel.initializer(attr.HeUniform())
    kernel.activation(attr.Relu())
    kernel.batch_normalization()
    kernel.max_pooling(shape=(2, 2), stride=2)

    config.flatten()
    dense = config.dense()

    layer = dense.add_layer(size=128)
    layer.initializer(attr.HeUniform())
    layer.activation(attr.Relu())

    layer = dense.add_layer(size=64)
    layer.initializer(attr.HeUniform())
    layer.activation(attr.Relu())

    layer = dense.add_layer(size=64, dropout=0.2)
    layer.initializer(attr.HeUniform())
    layer.activation(attr.Relu())

    layer = dense.add_layer(size=24, dropout=0.3)
    layer.initializer(attr.HeUniform())
    layer.activation(attr.Relu())

    # Output Layer
    output = dense.add_layer(size=1)
    output.activation(attr.Sigmoid())
    output.initializer(attr.XavierUniform())
    
    return config

def data_set() -> ImageProcessor:
    return ImageProcessor(
            base_dir="./data/breast-histopathology-images",
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            split_ratios=(0.80, 0.10, 0.10),
            shuffle=True,
            augmentation=True,
            augmentation_params={
                'rotation': 20,
                'zoom': 0.2,
                'horizontal_flip': True,
                'shear': 0.2,
                'fill_mode': 'nearest'
            }
        )

def create_app(config: Config) -> nn.App:
    return nn.app.App(
        model=config.new_model(),
        board=FileInput(
            title="Breast Cancer Recognition",
            img_resize=IMAGE_SIZE,
        )
    )

def train_model(app: nn.App, plot: bool):
    app.model().set_training_mode()
    app.model().get_trainer().train(
        processor=data_set(),
        epochs=EPOCHS,
        plot=Chart().plot_metrics if plot else None,
        callbacks=[attr.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7)]
    )

def validate_model(app: nn.App):
    app.draw()
    app.model().set_test_mode()
    app.board().set_handler(handler=app.predict_image)
    app.board().set_labels(path_json="./labels/breast_cancer.json")
    app.loop()


def test_model(app: nn.App, plot: bool):
    app.model().set_test_mode()
    app.model().get_tester().test(processor=data_set(), plot=Chart().plot_metrics if plot else None)


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
