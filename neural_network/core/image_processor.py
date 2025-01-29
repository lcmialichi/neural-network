import os
import random
from PIL import Image
from neural_network.gcpu import gcpu
from typing import Tuple, List, Generator
from neural_network.core.processor import Processor
from PIL import ImageEnhance 

class ImageProcessor(Processor):
    def __init__(self, 
        base_dir: str, 
        image_size: Tuple[int, int] = (64, 64), 
        batch_size: int = 32, 
        rotation_range: int = 30, 
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15), 
        shuffle: bool = True,
        rand_horizontal_flip: float = 0.0,
        rand_vertical_flip: float = 0.0,
        rand_brightness: float = 0.0,
        rand_contrast: float = 0.0,
        rand_crop: float = 0.0,
    ):
        self.base_dir = base_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.rotation_range = rotation_range
        self.split_ratios = split_ratios
        self.shuffle = shuffle
        self.rand_horizontal_flip = rand_horizontal_flip
        self.rand_vertical_flip = rand_vertical_flip
        self.rand_brightness = rand_brightness
        self.rand_contrast = rand_contrast
        self.rand_crop = rand_crop

        self.train_sample, self.validation_sample, self.test_sample = self._split_samples()

    def _split_samples(self) -> Tuple[List[str], List[str], List[str]]:
        """Splits patients into training, validation, and testing sets."""
        sample = [d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d))]

        train_size = int(self.split_ratios[0] * len(sample))
        val_size = int(self.split_ratios[1] * len(sample))
        train = sample[:train_size]
        validation = sample[train_size:train_size + val_size]
        test = sample[train_size + val_size:]

        return train, validation, test

    def _load_image(self, image_path: str, apply_mask: bool = False) -> gcpu.ndarray:
        """Loads an image from the path and applies transformations."""
        try:
            image = Image.open(image_path).convert('RGB')
            if apply_mask:
                image = self._apply_mask(image)

            img_data = gcpu.array(image.resize(self.image_size))
            img_data = gcpu.transpose(img_data, (2, 0, 1))
            return img_data
        except Exception as e:
            raise SystemError(f"Unable to process the image {image_path}: {e}")

    def _apply_mask(self, image): 
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        image = image.rotate(angle)

        if random.random() <= self.rand_horizontal_flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() <= self.rand_vertical_flip:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)

        if random.random() <= self.rand_brightness:
            factor = random.uniform(0.8, 1.2)
            image = ImageEnhance.Brightness(image).enhance(factor)

        if random.random() <= self.rand_crop:
            width, height = image.size
            crop_size = (random.uniform(0.8, 1.0) * width, random.uniform(0.8, 1.0) * height)
            left = random.uniform(0, width - crop_size[0])
            top = random.uniform(0, height - crop_size[1])
            image = image.crop((left, top, left + crop_size[0], top + crop_size[1]))
            image = image.resize(self.image_size)

        return image

    def _generate_batches(self, patient_paths: List[str], apply_mask: bool = False) -> Generator[Tuple[gcpu.ndarray, gcpu.ndarray], None, None]:
        """
        Generates batches by loading images on demand.

        :param patient_paths: List of patient directories.
        """
        all_image_paths = []
        for patient in patient_paths:
            patient_dir = os.path.join(self.base_dir, patient)
            for class_label in os.listdir(patient_dir):
                class_dir = os.path.join(patient_dir, class_label)
                if os.path.isdir(class_dir):
                    label = int(class_label)
                    for image_file in os.listdir(class_dir):
                        image_path = os.path.join(class_dir, image_file)
                        all_image_paths.append((image_path, label))

        random.shuffle(all_image_paths)

        batch_data = []
        batch_labels = []

        for image_path, label in all_image_paths:
            img = self._load_image(image_path, apply_mask)
            batch_data.append(img)
            batch_labels.append(gcpu.eye(2)[label])

            if len(batch_data) == self.batch_size:
                yield gcpu.array(batch_data), gcpu.array(batch_labels)
                batch_data, batch_labels = [], []

        if batch_data:
            yield gcpu.array(batch_data), gcpu.array(batch_labels)

    def get_train_batches(self) -> Generator[Tuple[gcpu.ndarray, gcpu.ndarray], None, None]:
        """Generates training batches."""
        if self.shuffle:
            random.shuffle(self.train_sample)
        return self._generate_batches(self.train_sample, apply_mask=True)

    def get_val_batches(self) -> Generator[Tuple[gcpu.ndarray, gcpu.ndarray], None, None]:
        """Generates validation batches."""
        return self._generate_batches(self.validation_sample, apply_mask=False)

    def get_test_batches(self) -> Generator[Tuple[gcpu.ndarray, gcpu.ndarray], None, None]:
        """Generates test batches."""
        return self._generate_batches(self.test_sample, apply_mask=False)
