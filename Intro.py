import os
import sys
import time
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
    page_title="Style Transfer Playground",
    page_icon="ℹ️",
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

st.title("Style Transfer Playground")
st.markdown('')
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
        update_streamlit=True,
        width=300,
        height=256,
        drawing_mode="freedraw",
        key="canvas_hardcoded",
        display_toolbar=False
    )

# track canvas state outside of the column
@st.cache
def get_rendering_tag(_canvas):
    unique_elements = np.unique(_canvas.image_data)
    if len(unique_elements) > 2:
        render_display_tile = True
    else:
        render_display_tile = False
    return render_display_tile

render_display_tile = get_rendering_tag(canvas_hardcoded)

with col2:
    st.write('...and see what happens')
    # Do something interesting with the image data and paths
    if render_display_tile:
    # if canvas_hardcoded.image_data is not None and len(unique_elements) > 2:

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
if render_display_tile:
# if canvas_hardcoded.image_data is not None and len(unique_elements) > 2:
    st.markdown(
        "You can draw some more on [**📃 Canvas**](/Canvas).",
        unsafe_allow_html=True
        )
    st.markdown(
        "Or, if you want to try style transfer on your own images, go straight to [**🎨 Style transfer**](/Style_transfer).",
        unsafe_allow_html=True
        )
