"""
    This file is a part of the Facial Emotion Recognition project
    @author Danofred00
    @author Jonas
"""

import cv2
import os
import numpy as np
import tensorflow as tf
from .constants import DEFAULT_RESHAPE

def image_to_pixel_seq(image: cv2.typing.MatLike) -> str:
    """
        This function is responsible to retrieve an array like image bytes
        and return all pixels of the image into a string
    """
    
    return ' '.join(map(str, image.flatten()))

def read_image(path: str) -> cv2.typing.MatLike:
    """
        This is responsible to read the image file and
        return a matrix of image's bytes
    """

    if(not os.path.exists(path)):
        raise RuntimeError(f"[-] Error while reading the image : path {path} not exists.")

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, DEFAULT_RESHAPE)

    return image


def transform_sequence(seq: str):
    """
        This is responsible to transform a sequence of pixels
        into an array the program can recognize
    """
    
    pixels = np.array([int(pixel) for pixel in seq.split()])
    image = pixels.reshape(48, 48)
    image_normalized = image.astype('float32') / 255.0
    image_normalized = np.expand_dims(image_normalized, axis=-1)
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    return image_batch


def load_model(path: str) -> tf.keras.models.Model:
    """
        This is responsible to load the pretrained model
    """
    
    model = tf.keras.models.load_model(path)
    return model