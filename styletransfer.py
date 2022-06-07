# %%
import os

import streamlit as st
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from util import crop_center, load_img_url, load_img_path, show_n, transform_img, tensor_to_image


# # Only use the below code if you have low resources.
# os.environ['CUDA_VISIBLE_DEVICES'] = ""
# os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

# # For supressing warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# streamlit config
st.set_page_config(
    page_title="Fuckery",
    page_icon="🎈",
)

# model config
@st.cache
def get_model():
    hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    hub_module = hub.load(hub_handle)
    return hub_module

model = get_model()


# width
def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )

_max_width_()

st.title("Style transfer fun zone")
st.header("")

with st.expander("ℹ️ - About this app", expanded=True):

    st.write(
        """
        Careful. I'm up to here guys.
	    """
    )
    st.markdown("")

# set up dividers
col1, col2 = st.columns(2)

with col1:
    st.write('Content')
    content_image_file = st.file_uploader(
        "Upload main image", type=("png", "jpg"))
    # try:
    content_image_file = content_image_file.read()
    content_image = transform_img(content_image_file)
    # except:
        # pass

with col2:
    st.write('Style')
    style_image_file = st.file_uploader(
        "Upload style image", type=("png", "jpg"))
    try:
        style_image_file = style_image_file.read()
        style_image = transform_img(style_image_file)
        style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')
    except:
        pass


# %%
# stylize
predict = st.button('Start Neural Style Transfer...')

if predict:
    outputs = model(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0]
    # show_n([content_image, style_image, stylized_image], titles=['Original content image', 'Style image', 'Stylized image'])
    final_img = tensor_to_image(stylized_image)
    st.image(final_img)

# %%
