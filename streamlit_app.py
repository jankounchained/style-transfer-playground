import os
import time
import pandas as pd
import numpy as np
from PIL import Image

import streamlit as st
from streamlit_drawable_canvas import st_canvas

import tensorflow as tf
import tensorflow_hub as hub

from util import transform_img, tensor_to_image, load_img_path, load_img_array


###
### backend
###

# streamlit config
st.set_page_config(
    page_title="Landing page",
    page_icon="👺",
)

# helper funs specific for this app
def convert_canvas(canvas_data):

    # convert from RGBA to RGB & save
    canvas_rgb = canvas_data[:, :, 0:3]
    canvas_img = Image.fromarray(canvas_rgb, "RGB")
    canvas_img.save("canvas.png")
    # load and transform
    read_img = tf.io.read_file("canvas.png")
    transformed_img = transform_img(read_img, max_dim=300)

    return transformed_img


def quick_style_transfer(content_image, style_image):
    outputs = model(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0]
    final_img = tensor_to_image(stylized_image)
    return final_img


# model config
@st.cache
def get_model():
    hub_handle = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
    hub_module = hub.load(hub_handle)
    return hub_module


# @st.cache
def get_style_ref_imgs(reference_dir="reference-img/"):

    paths = [os.path.join(reference_dir, path) for path in os.listdir(reference_dir)]

    style_image_collection = []
    for path in paths:
        style_image = load_img_path(path)
        style_image = tf.nn.avg_pool(
            style_image, ksize=[3, 3], strides=[1, 1], padding="SAME"
        )
        style_image_collection.append(style_image)

    return style_image_collection


model = get_model()
style_image_collection = get_style_ref_imgs()


###
### app layout
###

st.title("TITLE")
st.markdown('DESCRIPTION')
st.header("")

col1, col2 = st.columns([1, 1])

with col1:
    # Create a canvas component
    canvas_hardcoded = st_canvas(
        fill_color="rgba(255, 165, 0, 1)",
        stroke_width=5,
        stroke_color="#000000",
        background_color="#EEEEEE",
        # background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=True,
        width=300,
        height=256,
        drawing_mode="freedraw",
        # point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        key="canvas",
    )

with col2:
    # Do something interesting with the image data and paths
    if canvas_hardcoded.image_data is not None:

        random_picker = np.random.randint(0, len(style_image_collection))
        picked_style_img = style_image_collection[random_picker]
        transformed_canvas = convert_canvas(canvas_hardcoded.image_data)

        display_img = quick_style_transfer(
            content_image=transformed_canvas, style_image=picked_style_img
        )

        st.image(display_img)

st.markdown('')
