import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image


def crop_center(image):
    """Returns a cropped square image."""
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, new_shape, new_shape
    )
    return image


def load_img_url(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
    """Loads and preprocesses images."""
    # Cache image file locally.
    image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
    # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
    img = tf.io.decode_image(tf.io.read_file(image_path), channels=3, dtype=tf.float32)[
        tf.newaxis, ...
    ]

    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img


def load_img_path(image_path, image_size=(256, 256), preserve_aspect_ratio=True):
    img = tf.io.decode_image(tf.io.read_file(image_path), channels=3, dtype=tf.float32)[
        tf.newaxis, ...
    ]
    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img


def load_img_array(image_array, image_size=(256, 256), preserve_aspect_ratio=True):
    img = tf.io.decode_image(image_array, channels=3, dtype=tf.float32)[tf.newaxis, ...]
    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img


def transform_img(img, max_dim=512):
    img = tf.image.decode_image(img)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    # if max_dim > long_dim:
    #     raise ValueError('fuck you')
    # else:
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


def convert_canvas(canvas_data):

    # convert from RGBA to RGB & save
    canvas_rgb = canvas_data[:, :, 0:3]
    canvas_img = Image.fromarray(canvas_rgb, "RGB")
    canvas_img.save("canvas.png")
    # load and transform
    read_img = tf.io.read_file("canvas.png")
    transformed_img = transform_img(read_img, max_dim=300)

    return transformed_img
