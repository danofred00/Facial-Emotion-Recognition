"""
    This file is a part of the Facial Emotion Recognition project
    @author Danofred00
    @author Jonas
"""

from .utils import *
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def image_to_x(path: str):
    image = read_image(path)
    pixel_sequence = image_to_pixel_seq(image)
    X = transform_sequence(pixel_sequence)
    return X
