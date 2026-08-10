import streamlit as st
import pandas as pd
import numpy as np
import math
from sentence_transformers import SentenceTransformer
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR/"cleaned_movies_data_from_api.csv"
NPY_PATH = BASE_DIR/"movie_embeddings.npy"
@st.cache_resource

def load_assets():
    df = pd.read_csv(CSV_PATH)
    embeddings = np.load(NPY_PATH)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df['overview'] = df['overview'].fillna('')
    mask = ((df['overview'].str.len()>50) & (df['vote_average']>0.0)).to_numpy()
    cleaned_df =  df[mask].reset_index(drop = True)
    cleaned_embeddings = embeddings[mask]
    return cleaned_df, cleaned_embeddings, model

df , embeddings , model = load_assets()

st.title('🎥AI Semantic Movie Recommender')
st.write('Find awesome movies based on genre, vibe and story concepts!')
user_query = st.text_input("Enter the description of the movie of your taste:",
                           placeholder="e.g., terrifying, pyschological, jump-scare filled horror movie!")

top_k = st.slider("Number of movie options you want:",
                  min_value=3,max_value=10, value=5)
search_button = st.button("Find Recommendations")

if search_button and user_query:
    with st.spinner("Searching semanting space..."):
        query_vec = model.encode([user_query], convert_to_numpy = True)
        scores = cosine_similarity(query_vec,embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = df.iloc[top_indices].copy()
        results['match_score'] = np.round(scores[top_indices]*100,1)
        
        
        
    st.subheader('Top Matches for You:')
    for _, row in results.iterrows():
        year = str(row['release_date'])[:4] if pd.notnull(row.get('release_date')) else "N/A"
        st.markdown(f"### {row['title']} ({year}) - 🌟 {row['vote_average']:.1f}/10")
        st.caption(f"**Similarity Match:**{row['match_score']:.1f}%")
        
        st.write(row['overview'])
        st.divider()
        