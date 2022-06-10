from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from src.util import convert_canvas

# streamlit config
st.set_page_config(
    page_title="Style Transfer Playground",
    page_icon="📃",
)

st.markdown("Here you can draw. When you're done, you can use your drawing in **🎨 Style transfer**")

# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "point", "line", "rect", "circle", "transform")
)

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
# FIXME: disabling bg image upload, as this doesn't carry ove to style transfer
# bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

# Create a canvas component
def init_canvas():
    canvas_result = st_canvas(
        initial_drawing=initial_drawing,
        fill_color="rgba(255, 165, 0, 1)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=None,
        update_streamlit=True,
        width=512,
        height=512,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        key="canvas"
    )
    return canvas_result

# reset canvas, if desired
reset_canvas = st.sidebar.button("Reset canvas")
if reset_canvas:
    # remove local initial drawing
    initial_drawing = None
    # remove image data from session state
    if 'initial_drawing' in st.session_state:
        del st.session_state['initial_drawing']
    if 'canvas_img' in st.session_state:
        del st.session_state['canvas_img']
    if 'canvas_tensor' in st.session_state:
        del st.session_state['canvas_tensor']
    # rerun canvas
    st.experimental_rerun()

# reuse initial drawing, if available
if 'initial_drawing' in st.session_state:
    initial_drawing = st.session_state['initial_drawing']
else:
    initial_drawing = None

# init canvas 
canvas_result = init_canvas()

# send to style transfer
send_to_nst = st.button('Use for Style Transfer')

if send_to_nst:
    # transform img
    transformed_canvas = convert_canvas(canvas_result.image_data)
    # save in session state
    st.session_state['canvas_img'] = canvas_result.image_data
    st.session_state['canvas_tensor'] = transformed_canvas
    st.session_state['initial_drawing'] = canvas_result.json_data
    st.write('Ok, your drawing is now loaded. Go to **🎨 Style transfer**')
