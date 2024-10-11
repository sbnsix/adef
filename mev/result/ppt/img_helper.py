""" Image helper module"""


from PIL import Image
from pptx.util import Inches


class ImgHelper:
    @staticmethod
    def resize_image(
        self, image_path: str, x1: int, y1: int, x2: int, y2: int
    ) -> object:
        img = Image.open(image_path)
        width, height = img.size
        aspect_ratio = width / height
        new_width = x2 - x1
        new_height = y2 - y1
        new_ratio = new_width / new_height
        if new_ratio == aspect_ratio:
            img = img.resize((new_width, new_height), Image.ANTIALIAS)
        elif new_ratio > aspect_ratio:
            img = img.resize(
                (int(new_height * aspect_ratio), new_height), Image.ANTIALIAS
            )
            offset = (new_width - int(new_height * aspect_ratio)) // 2
            img = img.crop((-offset, 0, new_width - offset, new_height))
        else:
            img = img.resize(
                (new_width, int(new_width / aspect_ratio)), Image.ANTIALIAS
            )
            offset = (new_height - int(new_width / aspect_ratio)) // 2
            img = img.crop((0, -offset, new_width, new_height - offset))
        return img

    @staticmethod
    def get_image_dpi(image_path: str) -> float:
        with Image.open(image_path) as img:
            dpi = img.info.get("dpi")
        return dpi[0]

    @staticmethod
    def get_image_size(image_path: str) -> (int, int):
        width = 0
        height = 0
        with Image.open(image_path) as img:
            width, height = img.size

        return width, height

    @staticmethod
    def inches_to_pixels(inches: float, dpi: float):
        emus = Inches(inches)
        pixels = emus / 914400 * dpi
        return int(pixels)

    @staticmethod
    def px_to_inches(px: int, ppi: int = 96) -> float:
        """
        Convert pixels to inches.
        :param px: int, the number of pixels
        :param ppi: int, the pixels per inch value (default is 96 for web)
        :return: float, the equivalent value in inches
        """
        return px / ppi
