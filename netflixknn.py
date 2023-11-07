import streamlit as st
import numpy as np
from gensim.models.fasttext import FastText as FT_gensim
from sklearn.cluster import KMeans
import pandas as pd
import pickle
from joblib import load
import gdown
import os

def download_model(file_id, output_path):
    # Construct the gdown URL
    url = f'https://drive.google.com/uc?id={file_id}'
    
    # Download the file
    gdown.download(url, output_path, quiet=False)

# The ID of your file on Google Drive (just the ID, not the full URL)
FILE_ID = '1HXGBWtS-X6Mm6vdpNF1-QFh0KxljczC3'

# Check if the file does not already exist to avoid re-downloading it
if not os.path.exists('fasttext_model.pkl'):
    # Call the download function with the actual FILE_ID and the desired output path
    download_model(FILE_ID, 'fasttext_model.pkl')

# Now the model file should be in your local directory and can be loaded
with open('fasttext_model.pkl', 'rb') as ft_file:
    FT_model = pickle.load(ft_file)


# Custom CSS
def custom_css():
    st.markdown("""
    <style>
        .stApp {
            background-color: #000000;
        }
        ...
    </style>
    """, unsafe_allow_html=True)


# Load the model
loaded_kmean_model = load('kmeans_model.joblib')

# Load the dataset
@st.cache_data  # Using the new caching decorator
def load_data():
    df = pd.read_csv('processed_netflix_titles.csv')
    return df[['title', 'new_description', 'cluster_id']].dropna()

# Use the load_data function to load and cache your data
netflix = load_data()

# Define your recommendation function
def recommendation_system(title_name, top_k=5):
    title_row = netflix[netflix["title"] == title_name]
    if title_row.empty:
        return pd.DataFrame()
    
    search_df = netflix[netflix["cluster_id"].isin(title_row["cluster_id"])].copy()
    search_df = search_df.drop(search_df[search_df["title"] == title_name].index)
    
    def calculate_similarity(row):
        try:
            return FT_model.wv.similarity(title_row["new_description"].values[0], row["new_description"])
        except KeyError:
            return 0
    
    search_df["Similarity"] = search_df.apply(calculate_similarity, axis=1)
    search_df.sort_values(by=["Similarity"], ascending=False, inplace=True)
    
    return search_df[["title", "Similarity"]].head(top_k)

# Streamlit app

custom_css()

st.image('Logos-Readability-Netflix-logo.jpg', caption=None, width=None, use_column_width=True)
st.markdown("<h1 style='text-align: center; color: white;'>Netflix Movie Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white;'>Select a movie title you like to get recommendations for similar titles you might enjoy:</p>", unsafe_allow_html=True)

# Assuming 'netflix' is your DataFrame with movie titles after loading data and preprocessing
title_name = st.selectbox('Select a movie title:', netflix['title'].unique())

# When the button is pressed, make the recommendation
if st.button('Get Your Next Netflix Watch'):
    recommendations = recommendation_system(title_name)
    
    if recommendations.empty:
        st.markdown("<p style='text-align: center; color: white;'>Sorry, we couldn't find any recommendations for this title.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='text-align: center; color: white;'>Recommendations:</p>", unsafe_allow_html=True)
        # Iterate through the recommendations and display them
        for i, title in enumerate(recommendations['title'], start=1):  # Adjust field name if necessary
            st.markdown(f"<p style='text-align: left; color: white;'>{i}. {title}</p>", unsafe_allow_html=True)

st.image("Demi-Lovato-Popcorn-Gif.gif", caption="", use_column_width=True)

