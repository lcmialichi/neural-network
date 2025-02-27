import os
import random
from PIL import Image
from neural_network.gcpu import driver
from typing import Tuple, List, Generator
from neural_network.core.processor import Processor
from neural_network.core.augmentations import Augmentations
import glob

class ImageProcessor(Processor):
    def __init__(
        self, 
        base_dir: str, 
        image_size: Tuple[int, int] = (50, 50), 
        batch_size: int = 32, 
        shuffle: bool = True,
        split_ratios: tuple = (0.8, 0.1),
        augmentation: bool = False,
        augmentation_params: dict | None = None 
    ):
        if augmentation_params is None:
            augmentation_params = {}

        self.base_dir = base_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.augmentation_params = augmentation_params
        self.split_ratios = split_ratios
        self.load_samples()
        
    def load_samples(self):
        self.train_sample, self.validation_sample, self.test_sample = self._split_samples()   

    def _split_samples(self) -> Tuple[List[str], List[str], List[str]]:
        """Splits images into training, validation, and testing sets."""
        all_images = glob.glob(f'{self.base_dir}/**/*.png', recursive=True)
        random.shuffle(all_images)

        class_0, class_1 = [], []
        for image_path in all_images:
            label = self._get_label_from_path(image_path)
            if label == 0:
                class_0.append(image_path)
            else:
                class_1.append(image_path)

        all_balanced_images = class_0 + class_1
        random.shuffle(all_balanced_images)

        train_size = int(self.split_ratios[0] * len(all_balanced_images))
        val_size = int(self.split_ratios[1] * len(all_balanced_images))
        
        train_images = all_balanced_images[:train_size]
        validation_images = all_balanced_images[train_size:train_size + val_size]
        test_images = all_balanced_images[train_size + val_size:]
        return train_images, validation_images, test_images

    def _get_label_from_path(self, image_path: str) -> int:
        class_dir = os.path.basename(os.path.dirname(image_path))
        return int(class_dir)

    def _load_image(self, image_path: str, apply_mask: bool = False):
        """Loads an image from the path and applies transformations."""
        try:
            image = Image.open(image_path).convert('RGB')
            if self.augmentation and apply_mask:
                image = self._apply_augmentations(image)

            return driver.gcpu.array(image.resize(self.image_size))
        except Exception as e:
            raise SystemError(f"Unable to process the image {image_path}: {e}")

    def _generate_batches(self, image_paths: List[str], apply_mask: bool = False) -> Generator[Tuple, None, None]:
        batch_data, batch_labels = [], []
        for image_path in image_paths:
            label = self._get_label_from_path(image_path)
            img = self._load_image(image_path, apply_mask)
            batch_data.append(img)
            batch_labels.append(driver.gcpu.eye(2)[label])

            if len(batch_data) == self.batch_size:
                yield driver.gcpu.array(batch_data), driver.gcpu.array(batch_labels)
                batch_data, batch_labels = [], []

        if batch_data:
            yield driver.gcpu.array(batch_data), driver.gcpu.array(batch_labels)

    def get_train_batches(self) -> Generator[Tuple, None, None]:
        """Generates training batches."""
        if self.shuffle:
            random.shuffle(self.train_sample)
        return self._generate_batches(self.train_sample, apply_mask=True)

    def get_val_batches(self) -> Generator[Tuple, None, None]:
        """Generates validation batches."""
        return self._generate_batches(self.validation_sample, apply_mask=False)

    def get_test_batches(self) -> Generator[Tuple, None, None]:
        """Generates test batches."""
        return self._generate_batches(self.test_sample, apply_mask=False)

    def _apply_augmentations(self, image):

        augmentations = self._get_supported_augmentations()

        fill_modes = {
            'nearest': None,
            'constant': (0, 0, 0),
            'reflect': 'reflect',
            'wrap': 'wrap'
        }

        fill_value = fill_modes.get(self.augmentation_params.pop('fill_mode', 'nearest'), None)

        for key, value in self.augmentation_params.items():
            handler = augmentations.get(key)
            image = handler(image, value, fill_value)

        return image

    def _get_supported_augmentations(self):
        return {
            'rotation': Augmentations.rotation,
            'horizontal_flip': Augmentations.horizontal_flip,
            'vertical_flip': Augmentations.vertical_flip,
            'brightness': Augmentations.brightness,
            'contrast': Augmentations.contrast,
            'random_crop': Augmentations.random_crop,
            'blur': Augmentations.blur,
            'shear': Augmentations.shear,
            'zoom': Augmentations.zoom,
            'width_shift_range': Augmentations.width_shift_range,
            'height_shift_range': Augmentations.height_shift_range,
        }
