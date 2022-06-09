import os
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub

from src.util import tensor_to_image, load_img_path


def quick_style_transfer(model, content_image, style_image):
    outputs = model(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0]
    final_img = tensor_to_image(stylized_image)
    return final_img


@st.cache
def get_model():
    hub_handle = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
    hub_module = hub.load(hub_handle)
    return hub_module


def get_style_ref_imgs(reference_dir="reference-img/"):

    paths = [os.path.join(reference_dir, path) for path in os.listdir(reference_dir) if path.endswith('.jpg')]

    style_image_collection = []
    for path in paths:
        style_image = load_img_path(path)
        style_image = tf.nn.avg_pool(
            style_image, ksize=[3, 3], strides=[1, 1], padding="SAME"
        )
        style_image_collection.append(style_image)

    return style_image_collection
