import os
import random
from PIL import Image
from neural_network.gcpu import driver
from typing import Tuple, List, Generator
from neural_network.core.processor import Processor
from PIL import ImageEnhance, ImageFilter

class ImageProcessor(Processor):
    def __init__(
        self, 
        base_dir: str, 
        image_size: Tuple[int, int] = (64, 64), 
        batch_size: int = 32, 
        shuffle: bool = True,
        split_ratios: tuple = (0.7, 0.15, 0.15),
        augmentation: bool = False,
        augmentation_params: dict | None = None 
    ):
        default = {
            'rotation': 30,
            'horizontal_flip': 0.5,
            'vertical_flip': 0.5,
            'brightness': 0.2,
            'contrast': 0.2,
            'random_crop': 0.2,
            'blur': 0.1,
            'shear': 0.1,
            'zoom': 0.2,
    }

        if augmentation_params is not None:
            default.update(augmentation_params)

        self.base_dir = base_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.augmentation_params = default
        self.split_ratios = split_ratios
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

    def _load_image(self, image_path: str, apply_mask: bool = False):
        """Loads an image from the path and applies transformations."""
        try:
            image = Image.open(image_path).convert('RGB')
            if self.augmentation and apply_mask:
                image = self._apply_augmentations(image)

            img_data = driver.gcpu.array(image.resize(self.image_size))
            img_data = driver.gcpu.transpose(img_data, (2, 0, 1))
            return img_data
        except Exception as e:
            raise SystemError(f"Unable to process the image {image_path}: {e}")

    def _apply_augmentations(self, image):
        """Applies a series of augmentations based on the provided settings."""
        if random.random() <= self.augmentation_params.get('rotation'):
            angle = random.uniform(-self.augmentation_params.get('rotation'), self.augmentation_params.get('rotation'))
            image = image.rotate(angle)

        if random.random() <= self.augmentation_params.get('horizontal_flip'):
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() <= self.augmentation_params.get('vertical_flip'):
            image = image.transpose(Image.FLIP_TOP_BOTTOM)

        if random.random() <= self.augmentation_params.get('brightness'):
            factor = random.uniform(0.8, 1.2)
            image = ImageEnhance.Brightness(image).enhance(factor)

        if random.random() <= self.augmentation_params.get('contrast'):
            factor = random.uniform(0.8, 1.2)
            image = ImageEnhance.Contrast(image).enhance(factor)

        if random.random() <= self.augmentation_params.get('random_crop'):
            width, height = image.size
            crop_size = (random.uniform(0.8, 1.0) * width, random.uniform(0.8, 1.0) * height)
            left = random.uniform(0, width - crop_size[0])
            top = random.uniform(0, height - crop_size[1])
            image = image.crop((left, top, left + crop_size[0], top + crop_size[1]))
            image = image.resize(self.image_size)

        if random.random() <= self.augmentation_params.get('blur'):
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.5)))

        if random.random() <= self.augmentation_params.get('shear'):
            shear_factor = random.uniform(-0.3, 0.3)
            image = image.transform(
                image.size,
                Image.AFFINE,
                (1, shear_factor, 0, shear_factor, 1, 0),
                resample=Image.BICUBIC
            )
        
        if random.random() <= self.augmentation_params.get('zoom'):
            zoom_factor = random.uniform(1.0, 1 + self.augmentation_params.get('zoom'))
            width, height = image.size
            new_width, new_height = int(width * zoom_factor), int(height * zoom_factor)
            image = image.resize((new_width, new_height))
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            image = image.crop((left, top, left + width, top + height))

        return image

    def _generate_batches(self, patient_paths: List[str], apply_mask: bool = False) -> Generator[Tuple, None, None]:
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

