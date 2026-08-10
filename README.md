MOVIE-RECOMMENDER

This ml project is built using movies data from 'TmDB Api', sentence-transform(an encoder) and can be used through streamlit UI built using streamlit

While you are interacting with the model, you can describe the taste of the movie you wish to see, the better you describe it, the better the recommendation. One thing to keep in mind is that it will suggest you the number of movies you want(between 0 to 10), the top suggested movie can be of low rating but it will match you prompt better than the movies suggested at lower rank.


Project Structure:
     ML_PROJECT_01.ipynb contains the code used to fetch the data through APi.
     cleaned_movies_data_from_api.csv: generated through .ipynb file, contains movies data.
     movie_embeddings.npy: contains embeddings generated through sentence-transformers
     app.py: contains code that generated the UI of this project to be used by user, it also connect embeddings file and user input through cosine-similarity
     requirements.txt: contains the requirements of this project
     readme.md: Contains small description about the project
     


If you wish to use the model https://movierecommender-c2rhpnvuaxxtxo4gw4rlno.streamlit.app/
