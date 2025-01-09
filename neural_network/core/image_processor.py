import os
import random
from PIL import Image
import numpy as np
from typing import Tuple, List, Generator
from neural_network.core.processor import Processor

class ImageProcessor(Processor):
    def __init__(self, 
                 base_dir: str, 
                 image_size: Tuple[int, int] = (64, 64), 
                 batch_size: int = 32, 
                 rotation_range: int = 30, 
                 split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15), 
                 shuffle: bool = True):
        """
        :param base_dir: Root directory containing data organized by patient.
        :param image_size: Size to resize the images.
        :param batch_size: Batch size for training.
        :param rotation_range: Maximum angle for random rotation.
        :param split_ratios: Proportion for splitting into training, validation, and testing sets.
        :param shuffle: If True, randomizes the order of patients before splitting the data.
        """
        self.base_dir = base_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.rotation_range = rotation_range
        self.split_ratios = split_ratios
        self.shuffle = shuffle

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

    def _load_image(self, image_path: str) -> np.ndarray:
        """Loads an image from the path and applies transformations."""
        try:
            image = Image.open(image_path).convert('RGB')

            # Random rotation
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            image = image.rotate(angle)

            # Resize and normalize to CHW format
            img_data = np.array(image.resize(self.image_size))
            img_data = np.transpose(img_data, (2, 0, 1))
            return img_data
        except Exception as e:
            raise SystemError(f"Unable to process the image {image_path}: {e}")

    def _generate_batches(self, patient_paths: List[str]) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
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
                    label = int(class_label)  # Class 0 or 1
                    for image_file in os.listdir(class_dir):
                        image_path = os.path.join(class_dir, image_file)
                        all_image_paths.append((image_path, label))

        random.shuffle(all_image_paths)

        batch_data = []
        batch_labels = []

        for image_path, label in all_image_paths:
            img = self._load_image(image_path)
            batch_data.append(img)
            batch_labels.append(np.eye(2)[label])  # One-hot encoding

            if len(batch_data) == self.batch_size:
                yield np.array(batch_data), np.array(batch_labels)
                batch_data, batch_labels = [], []

        if batch_data:
            yield np.array(batch_data), np.array(batch_labels)

    def get_train_batches(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generates training batches."""
        if self.shuffle:
            random.shuffle(self.train_sample)
        return self._generate_batches(self.train_sample)

    def get_val_batches(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generates validation batches."""
        return self._generate_batches(self.validation_sample)

    def get_test_batches(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generates test batches."""
        return self._generate_batches(self.test_sample)
