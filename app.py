import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load your dataset and save it to a pickle file
movies_df = pd.read_csv('Movie_database.csv')
movies_df.to_pickle("movie_list.pkl")

# Read the DataFrame from the pickle file
movies = pd.read_pickle("movie_list.pkl")

# Content-Based Filtering
cv = CountVectorizer()
genre_matrix = cv.fit_transform(movies['Genre'])
features_matrix = genre_matrix

# Calculate cosine similarity
content_similarity = cosine_similarity(features_matrix, features_matrix)

# Function to get content-based recommendations
def get_content_based_recommendations(movie_title):
    # Convert the input movie title to lowercase
    movie_title_lower = movie_title.lower()

    # Check if the movie title exists in the dataset
    if movie_title_lower not in movies['Title'].str.lower().values:
        return pd.DataFrame(columns=['Title', 'Platform', 'Imdb'])  # Return an empty DataFrame

    movie_index = movies.index[movies['Title'].str.lower() == movie_title_lower].tolist()[0]
    sim_scores = list(enumerate(content_similarity[movie_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Exclude the movie itself, take top 5 similar movies
    movie_indices = [i[0] for i in sim_scores]
    recommendations = movies[['Title', 'Platform', 'Imdb']].iloc[movie_indices]

    # Check for repeated movies and select unique ones
    unique_recommendations = recommendations.drop_duplicates(subset='Title', keep='first')

    return unique_recommendations

# Hybrid Recommendation
def hybrid_recommendation(movie_title):
    # Get content-based recommendations
    content_based_recommendations = get_content_based_recommendations(movie_title)

    if content_based_recommendations.empty:
        return pd.DataFrame(columns=['Title', 'Platform', 'Imdb']), None  # Return empty DataFrame and None

    # Calculate platform distribution
    platform_distribution = content_based_recommendations['Platform'].value_counts()

    # Exclude the input movie from the IMDb rating comparison
    input_movie_imdb_rating = movies.loc[movies['Title'].str.lower() == movie_title.lower(), 'Imdb'].iloc[0]
    content_based_recommendations = content_based_recommendations[content_based_recommendations['Imdb'] != input_movie_imdb_rating]

    # Check if the highest IMDb rating is associated with the input movie
    if input_movie_imdb_rating == content_based_recommendations['Imdb'].max():
        # Choose the platform with the next highest IMDb rating
        recommended_platform = content_based_recommendations.sort_values('Imdb', ascending=False).iloc[1]['Platform']
    else:
        # Select the platform with the highest frequency
        recommended_platform = platform_distribution.idxmax()

    return content_based_recommendations, recommended_platform

# Streamlit app
st.header('Movie Recommender System')
selected_movie = st.selectbox('Select Movie From Dropdown', movies['Title'])

if st.button('Show Recommendations'):
    recommendations, recommended_platform = hybrid_recommendation(selected_movie)

    st.write(f"Content-Based Recommendations for {selected_movie}:")
    st.table(recommendations)

    st.write(f"Recommended Platform for movies like {selected_movie}: {recommended_platform}")
