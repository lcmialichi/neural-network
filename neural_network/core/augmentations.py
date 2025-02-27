from PIL import Image, ImageEnhance, ImageFilter
import random

class Augmentations:
    @staticmethod
    def rotation(image, arg, fill_mode):
        angle = random.uniform(-arg, arg)
        return image.rotate(angle, resample=Image.BICUBIC, fillcolor=fill_mode if isinstance(fill_mode, tuple) else None)

    @staticmethod
    def horizontal_flip(image, arg, fill_mode):
        if arg and random.random() < 0.5:
            return image.transpose(Image.FLIP_LEFT_RIGHT)

        return image

    @staticmethod
    def vertical_flip(image, arg, fill_mode):
        if arg and random.random() < 0.5:
            return image.transpose(Image.FLIP_TOP_BOTTOM)
        
        return image
    
    @staticmethod
    def brightness(image, arg, fill_mode):
        factor = random.uniform(1 - arg, 1 + arg)
        return ImageEnhance.Brightness(image).enhance(factor)
    
    @staticmethod
    def contrast(image, arg, fill_mode):
        factor = random.uniform(1 - arg, 1 + arg)
        return ImageEnhance.Contrast(image).enhance(factor)

    @staticmethod
    def random_crop(image, arg, fill_mode):
        width, height = image.size
        crop_size = (int((1 - arg) * width), int((1 - arg) * height))
        left = random.randint(0, width - crop_size[0])
        top = random.randint(0, height - crop_size[1])
        image = image.crop((left, top, left + crop_size[0], top + crop_size[1]))
        return image.resize((width, height))

    @staticmethod
    def blur(image, arg, fill_mode):
        return image.filter(ImageFilter.GaussianBlur(radius=arg))

    @staticmethod
    def shear(image, arg, fill_mode):
        shear_factor = random.uniform(-arg, arg)
        image = image.transform(
            image.size,
            Image.AFFINE,
            (1, shear_factor, 0, shear_factor, 1, 0),
            resample=Image.BICUBIC,
            fillcolor=fill_mode if isinstance(fill_mode, tuple) else None
        )
        
        return image

    @staticmethod
    def zoom(image, arg, fill_mode):
        zoom_factor = 1 + random.uniform(0, arg)
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
    
    @staticmethod
    def width_shift_range(image, arg, fill_mode):
        width, _ = image.size
        shift = int(random.uniform(-arg, arg) * width)
        return image.transform(
            image.size,
            Image.AFFINE,
            (1, 0, shift, 0, 1, 0),
            resample=Image.BICUBIC,
            fillcolor=fill_mode if isinstance(fill_mode, tuple) else None
        )

    @staticmethod
    def height_shift_range(image, arg, fill_mode):
        _, height = image.size
        shift = int(random.uniform(-arg, arg) * height)
        return image.transform(
            image.size,
            Image.AFFINE,
            (1, 0, 0, 0, 1, shift),
            resample=Image.BICUBIC,
            fillcolor=fill_mode if isinstance(fill_mode, tuple) else None
        )