import argparse
import neural_network as nn
import neural_network.supply as attr

from neural_network import Config
from neural_network.board import Chart, FileInput
from neural_network.core.padding import Padding
from neural_network.core.image_processor import ImageProcessor
from neural_network.console import Commands
from custom.residual_block import ResidualBlock

IMAGE_SIZE = (50, 50)
IMAGE_CHANNELS = 3
BATCH_SIZE = 8
EPOCHS = 100

def create_configuration():
    config = Config()
    config.set_processor(
        ImageProcessor(
            base_dir="./data/breast-histopathology-images/IDC_regular_ps50_idx5",
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            split_ratios=(0.75, 0.15, 0.10),
            shuffle=True,
            augmentation=True,
            augmentation_params={
                'rotation': 30,
                'zoom': 0.2,
                'horizontal_flip': 0.5,
                'vertical_flip': 0.5,
                'brightness': 0.5,
                'contrast': 0.5,
                'shear': 0.3
            }
        )
    )

    config.driver('cpu')
    config.set_global_optimizer(attr.Adam(learning_rate=0.0005, weight_decay=1e-4))
    config.with_cache(path='./data/cache/model.pkl')
    config.padding_type(Padding.SAME)
    config.loss_function(attr.CrossEntropyLoss())

    # ---- Convolutional  ----
    kernel = config.add_kernel(number=64, shape=(3, 3), stride=1)
    kernel.initializer(attr.He())
    kernel.activation(attr.Relu())
    kernel.batch_normalization()
    
    residual = ResidualBlock(number=64, shape=(3, 3), stride=1)
    residual.max_pooling(shape=(2, 2), stride=2)
    config.add(residual)

    kernel = config.add_kernel(number=128, shape=(3, 3), stride=1)
    kernel.initializer(attr.He())
    kernel.activation(attr.Relu())
    kernel.batch_normalization()
    
    residual = ResidualBlock(number=128, shape=(3, 3), stride=1)
    residual.max_pooling(shape=(2, 2), stride=2)
    config.add(residual)
    
    kernel = config.add_kernel(number=256, shape=(3, 3), stride=1)
    kernel.initializer(attr.He())
    kernel.activation(attr.Relu())
    kernel.batch_normalization()
    
    residual = ResidualBlock(number=256, shape=(3, 3), stride=1)
    residual.max_pooling(shape=(2, 2), stride=2)
    config.add(residual)
     
    kernel = config.add_kernel(number=512, shape=(3, 3), stride=1)
    kernel.initializer(attr.He())
    kernel.activation(attr.Relu())
    kernel.batch_normalization()
    
    residual = ResidualBlock(number=512, shape=(3, 3), stride=1)
    residual.max_pooling(shape=(2, 2), stride=2)
    config.add(residual)

    # Flatten
    config.flatten()
    dense = config.dense()

    layer1 = dense.add_layer(size=1024, dropout=0.5)
    layer1.initializer(attr.He())
    layer1.activation(attr.LeakyRelu())

    layer2 = dense.add_layer(size=512, dropout=0.3)
    layer2.initializer(attr.He())
    layer2.activation(attr.LeakyRelu())

    # Output Layer
    output = dense.add_layer(size=2)
    output.activation(attr.Softmax())
    output.initializer(attr.Xavier())

    return config


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
        epochs=EPOCHS,
        plot=Chart().plot_metrics if plot else None,
        scheduler=attr.ReduceLROnPlateau(factor=0.3, patience=2, min_lr=1e-7)
    )


def validate_model(app: nn.App):
    app.draw()
    app.model().set_test_mode()
    app.board().set_handler(handler=app.predict_image)
    app.board().set_labels(path_json="./labels/breast_cancer.json")
    app.loop()


def test_model(app: nn.App, plot: bool):
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
