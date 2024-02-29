import torch
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import streamlit as st
import pandas as pd
import numpy as np
import pickle


#For embedding images, we need the non-multilingual CLIP model
img_model = SentenceTransformer('clip-ViT-B-32')
multi_lingual_model = SentenceTransformer('clip-ViT-B-32-multilingual-v1')


img_names = list(glob.glob('D_mart_Images/*.jpg') )

print("Images:", len(img_names))
# img_emb = img_model.encode([Image.open(filepath) for filepath in img_names], batch_size=128, convert_to_tensor=True, show_progress_bar=True)
# pickle.dump(img_emb,open('pkl_Dmart_image_embeddings.pkl','wb'))


embeddings = np.array(pickle.load(open('pkl_Dmart_image_embeddings.pkl','rb')))
print("IMage Embeddings Shape --> ",embeddings.shape)


def search_imgs_based_on_query(query, k=10):

    query_emb = multi_lingual_model.encode([query], convert_to_tensor=True, show_progress_bar=True)
    hits = util.semantic_search(query_emb, embeddings, top_k=k)[0]

    st.write("Query:")
    st.write(query)
    
    for hit in hits:
        st.write(img_names[hit['corpus_id']])
        st.image((img_names[hit['corpus_id']]), width=300)


def imge_to_imge(query_image_path, k=8):
    query_emb = img_model.encode([Image.open(query_image_path)], convert_to_tensor=True, show_progress_bar=True)
    hits = util.semantic_search(query_emb, embeddings, top_k=k)[0]

    st.write("Query:")
    # st.write(query_image_path)
    st.image(query_image_path, width=300)

    
    for hit in hits:
        st.write(img_names[hit['corpus_id']])
        # st.write(query_image_path)
        st.image((img_names[hit['corpus_id']]), width=300)


        
        
# img_to_img_search(Image.open(os.path.join(img_folder, rand_imge)), k=30)