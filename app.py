import io
import os
import streamlit as st
import requests
from PIL import Image

st.title('Image-Caption-Generator')
img_url = st.text_input(label='Enter an Image URL')

st.markdown('<center style="opacity: 70%">OR</center>', unsafe_allow_html=True)
img_upload = st.file_uploader(label='Upload Image', type=['jpg', 'png', 'jpeg'])
