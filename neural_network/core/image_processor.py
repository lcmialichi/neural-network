import os
import random
from PIL import Image
import numpy as np
from typing import Tuple, Generator

class ImageProcessor:
    def __init__(self, base_dir: str, image_size: Tuple[int, int] = (64, 64), batch_size: int = 32, rotation_range: int = 30):
        self.base_dir = base_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.rotation_range = rotation_range

    def _load_image(self, image_path: str) -> np.ndarray:
        try:
            image = Image.open(image_path).convert('RGB')
            
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            image = image.rotate(angle)

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

        class_data = [(label, images) for label, images in class_paths.items()]
        random.shuffle(class_data)

        all_images = []
        for label, images in class_data:
            all_images.extend([(img, label) for img in images])
        
        random.shuffle(all_images)

        batch_data = []
        batch_labels = []
        
        for img, label in all_images:
            batch_data.append(img)
            batch_labels.append(np.eye(num_classes)[label])
            
            if len(batch_data) == self.batch_size:
                yield np.array(batch_data), np.array(batch_labels)
                batch_data, batch_labels = [], []

        if batch_data:
            yield np.array(batch_data), np.array(batch_labels)

    def _process_directory(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        for sample_folder in os.listdir(self.base_dir):
            sample_path = os.path.join(self.base_dir, sample_folder)

            if not os.path.isdir(sample_path):
                raise ValueError(f"{sample_path} não é um diretório")

            yield from self._process_sample_folder(sample_path)

    def process_images(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        return self._process_directory()
