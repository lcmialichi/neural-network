import os
import random
from PIL import Image
import numpy as np
from typing import Tuple, Generator

class ImageProcessor:
    def __init__(self, base_dir: str, image_size: Tuple[int, int] = (64, 64), batch_size: int = 32):
        self.base_dir = base_dir
        self.image_size = image_size
        self.batch_size = batch_size

    def _load_image(self, image_path: str) -> np.ndarray:
        try:
            image = Image.open(image_path).convert('RGB')
            img_data = np.array(image.resize(self.image_size))
            img_data = np.transpose(img_data, (2, 0, 1)) 
            return img_data
        except Exception as e:
            raise SystemError(f"Não foi possível processar a imagem {image_path}: {e}")

    def _load_images_from_class(self, class_path: str) -> list:
        images = []
        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            images.append(self._load_image(image_path))
        return images

    def _process_sample_folder(self, sample_folder: str) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        class_paths = {}
        
        for class_folder in os.listdir(sample_folder):
            class_path = os.path.join(sample_folder, class_folder)
            if os.path.isdir(class_path):
                class_label = int(class_folder)
                class_paths[class_label] = self._load_images_from_class(class_path)

        num_classes = len(class_paths)
        if num_classes == 0:
            raise ValueError(f"Nenhuma classe encontrada em {sample_folder}")

        class_indices = {label: 0 for label in class_paths}
        while any(idx < len(images) for idx, images in zip(class_indices.values(), class_paths.values())):
            batch_data = []
            batch_labels = []

            for label, images in class_paths.items():
                start_idx = class_indices[label]
                end_idx = start_idx + (self.batch_size // num_classes)

                batch_class_data = images[start_idx:end_idx]

                if len(batch_class_data) < (self.batch_size // num_classes):
                    batch_class_data += random.choices(
                        images,
                        k=(self.batch_size // num_classes - len(batch_class_data))
                    )

                batch_class_labels = [np.eye(num_classes)[label]] * len(batch_class_data)
                batch_data.extend(batch_class_data)
                batch_labels.extend(batch_class_labels)

                class_indices[label] += (self.batch_size // num_classes)

            combined = list(zip(batch_data, batch_labels))
            random.shuffle(combined)
            batch_data, batch_labels = zip(*combined)

            yield np.array(batch_data), np.array(batch_labels)

    def _process_directory(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        for sample_folder in os.listdir(self.base_dir):
            sample_path = os.path.join(self.base_dir, sample_folder)

            if not os.path.isdir(sample_path):
                raise ValueError(f"{sample_path} não é um diretório")

            yield from self._process_sample_folder(sample_path)

    def process_images(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        return self._process_directory()
