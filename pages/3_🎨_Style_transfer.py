import os

import streamlit as st
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from src.util import crop_center, load_img_url, load_img_path, show_n, transform_img, tensor_to_image
from src.nst_config import get_model, get_style_ref_imgs, quick_style_transfer

# streamlit config
st.set_page_config(
    page_title="Style transfer",
    page_icon="🎨",
)

# model config
if 'nst_model' not in st.session_state:
    model = get_model()
    st.session_state['nst_model'] = model
else:
    model = st.session_state['nst_model']


st.title("Style transfer fun zone")
st.header("")

with st.expander("ℹ️ - Tutorial", expanded=False):

    st.write(
        """
        You need two images for Style Transfer:  
        A **content image** for objects and a **style image** for colors, textures and so on.   
        The result is an image, that has things from the content image, but looks like the style image.
	    """
    )
    st.markdown("")

# set up dividers
st.markdown("")
col1_a, col2_a = st.columns(2)

with col1_a:
    st.write('Content')
    content_image_file = st.file_uploader(
        "Upload main image", type=("png", "jpg"))

with col2_a:
    st.write('Style')
    style_image_file = st.file_uploader(
        "Upload style image", type=("png", "jpg"))


# show loaded images
col1_a2, col2_a2 = st.columns(2)
with col1_a2:
    if content_image_file or 'canvas_tensor' in st.session_state:
        st.write('Selected image')

    if content_image_file and not 'canvas_tensor' in st.session_state:
        content_image_file = content_image_file.read()
        content_image = transform_img(content_image_file)
        st.image(content_image_file)
    
    if 'canvas_tensor' in st.session_state and not content_image_file:
        content_image = st.session_state['canvas_tensor']
        st.image(st.session_state['canvas_img'])

with col2_a2:
    if style_image_file:
        st.write('Selected image')
        style_image_file = style_image_file.read()
        style_image = transform_img(style_image_file)
        style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')
        st.image(style_image_file)

# stylize
st.header('')
col1_b, col2_b, col3_b = st.columns([1, 0.5, 1])

with col2_b:
    predict = st.button('Transfer Style')


st.header('')
col1_c, col2_c, col3_c = st.columns([0.2, 5, 0.2])

with col2_c:
    if predict:
        st.write('Generated image')
        outputs = model(tf.constant(content_image), tf.constant(style_image))
        stylized_image = outputs[0]
        # show_n([content_image, style_image, stylized_image], titles=['Original content image', 'Style image', 'Stylized image'])
        final_img = tensor_to_image(stylized_image)
        st.image(final_img, use_column_width=True)




# # sidebar for image forslag
# from PIL import Image

# example = Image.open('examples/0.png')

# ahoj = st.button(st.image(example))

# if ahoj:
#     st.write('it works!')