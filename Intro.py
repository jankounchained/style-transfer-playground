import os
import sys
import time
import pandas as pd
import numpy as np
from PIL import Image

import streamlit as st
from streamlit_drawable_canvas import st_canvas

import tensorflow as tf

from src.util import transform_img, convert_canvas
from src.nst_config import get_model, get_style_ref_imgs, quick_style_transfer


###
### backend
###

# streamlit config
st.set_page_config(
    page_title="Van Gogh on a swing",
    page_icon="👺",
)


# model config
if 'nst_model' not in st.session_state:
    model = get_model()
    st.session_state['nst_model'] = model
else:
    model = st.session_state['nst_model']

style_image_collection = get_style_ref_imgs()



###
### app layout
###

st.title("Van Gogh on a Swing")
st.markdown('DESCRIPTION')
st.header("")

col1, col2 = st.columns([1, 1])

with col1:
    st.write('Try drawing...')
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

    unique_elements = np.unique(canvas_hardcoded.image_data)

with col2:
    st.write('...and see what happens')
    # Do something interesting with the image data and paths
    if canvas_hardcoded.image_data is not None and len(unique_elements) > 2:

        random_picker = np.random.randint(0, len(style_image_collection))
        picked_style_img = style_image_collection[random_picker]
        transformed_canvas = convert_canvas(canvas_hardcoded.image_data)

        display_img = quick_style_transfer(
            model=model,
            content_image=transformed_canvas,
            style_image=picked_style_img
        )

        st.image(display_img)

st.markdown('')
