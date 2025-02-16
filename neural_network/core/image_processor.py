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
            'random_crop': 0.0,
            'blur': 0.0,
            'shear': 0.1,
            'zoom': 0.2,
            'fill_mode': 'nearest'
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

    def _generate_batches(self, patient_paths: List[str], apply_mask: bool = False) -> Generator[Tuple, None, None]:
        """
        Gera batches balanceados entre múltiplas classes.
        """
        class_samples = {}

        # Separar as imagens em classes
        for patient in patient_paths:
            patient_dir = os.path.join(self.base_dir, patient)
            for class_label in os.listdir(patient_dir):
                class_dir = os.path.join(patient_dir, class_label)
                if os.path.isdir(class_dir):
                    label = int(class_label)
                    if label not in class_samples:
                        class_samples[label] = []
                    for image_file in os.listdir(class_dir):
                        image_path = os.path.join(class_dir, image_file)
                        class_samples[label].append((image_path, label))

        min_class_size = min(len(samples) for samples in class_samples.values())

        balanced_samples = []
        for label, samples in class_samples.items():
            if len(samples) > min_class_size:
                balanced_samples.extend(random.sample(samples, min_class_size))
            else:
                balanced_samples.extend(samples * (min_class_size // len(samples)) + random.sample(samples, min_class_size % len(samples)))

        random.shuffle(balanced_samples)

        batch_data, batch_labels = [], []
        num_classes = len(class_samples)

        for image_path, label in balanced_samples:
            img = self._load_image(image_path, apply_mask)
            batch_data.append(img)
            batch_labels.append(driver.gcpu.eye(num_classes)[label])
            if len(batch_data) == self.batch_size:
                yield driver.gcpu.array(batch_data), driver.gcpu.array(batch_labels)
                batch_data, batch_labels = [], []

        if batch_data:
            yield driver.gcpu.array(batch_data), driver.gcpu.array(batch_labels)

    def _apply_augmentations(self, image):
        """Applies a series of augmentations based on the provided settings."""
        params = self.augmentation_params
        fill_mode = params.get('fill_mode', 'nearest')
        fill_modes = {
            'nearest': None,
            'constant': (0, 0, 0),
            'reflect': 'reflect',
            'wrap': 'wrap'
        }
        fill_value = fill_modes.get(fill_mode, None)
        
        if 'rotation' in params:
            angle = random.uniform(-params['rotation'], params['rotation'])
            image = image.rotate(angle, resample=Image.BICUBIC, fillcolor=fill_value if isinstance(fill_value, tuple) else None)
        
        if params['horizontal_flip'] and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        if params['vertical_flip'] and random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        
        if 'brightness' in params:
            factor = random.uniform(1 - params['brightness'], 1 + params['brightness'])
            image = ImageEnhance.Brightness(image).enhance(factor)
        
        if 'contrast' in params:
            factor = random.uniform(1 - params['contrast'], 1 + params['contrast'])
            image = ImageEnhance.Contrast(image).enhance(factor)
        
        if 'random_crop' in params:
            width, height = image.size
            crop_size = (int((1 - params['random_crop']) * width), int((1 - params['random_crop']) * height))
            left = random.randint(0, width - crop_size[0])
            top = random.randint(0, height - crop_size[1])
            image = image.crop((left, top, left + crop_size[0], top + crop_size[1]))
            image = image.resize(self.image_size)
        
        if 'blur' in params:
            image = image.filter(ImageFilter.GaussianBlur(radius=params['blur']))
        
        if 'shear' in params:
            shear_factor = random.uniform(-params['shear'], params['shear'])
            image = image.transform(
                image.size,
                Image.AFFINE,
                (1, shear_factor, 0, shear_factor, 1, 0),
                resample=Image.BICUBIC,
                fillcolor=fill_value if isinstance(fill_value, tuple) else None
            )
        
        if 'zoom' in params:
            zoom_factor = 1 + random.uniform(0, params['zoom'])
            width, height = image.size
            new_width, new_height = int(width * zoom_factor), int(height * zoom_factor)
            image = image.resize((new_width, new_height), resample=Image.BICUBIC)
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            image = image.crop((left, top, left + width, top + height))
            
            if fill_mode in ['constant', 'reflect', 'wrap']:
                background = Image.new("RGB", (width, height), (0, 0, 0))
                background.paste(image, (0, 0))
                image = background
        
        return image