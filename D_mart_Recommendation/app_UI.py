import streamlit as st
from utils import *


st.title("text_to_image_search")
st.sidebar.text("Powered By U TRIPATHI 😎")

input_text = st.text_input("Enter name or brand")


submit_button =  st.button("Submit ")

if submit_button:
    st.write("Searching for images based on the query...")
    search_imgs_based_on_query(input_text)
    

st.title("Image-to-Image_Recommendation")
# upload_button = st.button("Upload image")

# if upload_button:
uploadded_file = st.file_uploader("upload yor Image .jpg" , type = ['jpg'])
if uploadded_file:
    imge_to_imge(uploadded_file)
else :
    st.write("please upload jpg file")
    


# 
