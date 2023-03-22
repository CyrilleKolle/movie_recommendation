from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process
import seaborn as sns
import matplotlib.pyplot as plt
import plotly_express as px
import imdb

app = Flask(__name__)
CORS(app)



movies, ratings = pd.read_csv('./data/movies.csv'), pd.read_csv('./data/ratings.csv')
movies['year']  = movies['title'].str.extract(r'\((\d{4})\)')
genres_df = movies['genres'].str.get_dummies('|')
movies['most_common_genre'] = genres_df.apply(lambda x: x.idxmax(), axis=1)
movies = movies.dropna(subset=['year'], how='any')
movies['year'] = movies['year'].astype(int)
movies.loc[:, 'title_no_year'] = movies['title'].apply(lambda x: x.split("(")[0].rstrip())

# new_ratings = ratings[ratings['movieId'].isin(movies['movieId'])]
# movieIds = pd.Categorical(new_ratings['movieId'], categories=movies['movieId'])
# userIds = pd.Categorical(new_ratings['userId'])

movieIds = pd.Categorical(ratings['movieId'])
userIds = pd.Categorical(ratings['userId'])

# Create the csr matrix
matrix = csr_matrix((ratings['rating'], (movieIds.codes, userIds.codes)))

model_KNN = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=22)
model_KNN.fit(matrix)


def recommender_system(movie_name, dataframe, model, number_recommendations):
    movie_id = process.extractOne(movie_name, movies['title'])[1]
    movie_idx = process.extractOne(movie_name, movies['title'])[2]
    print('Movie Selected: ', movies['title'][movie_idx], 'Id: ',movie_id)
    print('Searching for recommendation....')

    
    distances, indices = model.kneighbors(dataframe[movie_idx], n_neighbors=number_recommendations)

    indice = indices[0]
    selected = indice[indice != movie_idx]
    selected_movies = movies.iloc[selected]
    
    return selected_movies

def searched_movie(movie_name):
    movie = process.extractOne(movie_name, movies['title'])[0]
    movie_df = movies[movies['title'] == movie]
    return movie_df



@app.route('/api/dictionary')
def dictionary():
    word = request.args.get('word')
    # word_movie = movies[movies['title_no_year'] == word]
    
    recommendations = recommender_system(word, matrix, model_KNN, 22)
    searched = searched_movie(word)
    
    recommendations_dict = recommendations.to_dict(orient='records')
    searched_dict = searched.to_dict(orient='records')

    # return jsonify([recommendations[0].to_dict(orient='records'), recommendations[1].to_dict(orient='records')])
    return jsonify([recommendations_dict, searched_dict])

if __name__ == '__main__':
    app.run(debug=True, port=5100)