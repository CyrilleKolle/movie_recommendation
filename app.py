from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process
import seaborn as sns
import matplotlib.pyplot as plt
import plotly_express as px

app = Flask(__name__)
CORS(app)



movies, ratings = pd.read_csv('./data/movies.csv'), pd.read_csv('./data/ratings.csv')
movies['year']  = movies['title'].str.extract(r'\((\d{4})\)')
genres_df = movies['genres'].str.get_dummies('|')
movies['most_common_genre'] = genres_df.apply(lambda x: x.idxmax(), axis=1)
movies = movies.dropna(subset=['year'], how='any')
movies['year'] = movies['year'].astype(int)
movies.loc[:, 'title_no_year'] = movies['title'].apply(lambda x: x.split("(")[0].rstrip())

# movies['year']  = movies['title'].str.extract(r'\((\d{4})\)')
# genres_df = movies['genres'].str.get_dummies('|')
# movies['most_common_genre'] = genres_df.apply(lambda x: x.idxmax(), axis=1)
# movies = movies.dropna(subset=['year'], how='any')
# movies.loc[:, 'title_no_year'] = movies['title'].apply(lambda x: x.split("(")[0].rstrip())

# counts = ratings.groupby('userId')['movieId'].nunique()
# ratings = ratings[~ratings['userId'].isin(counts[counts > 1000].index)]
# # ratings = ratings[(ratings['rating'] >= 2) & (ratings['rating'] < 5)]
# new_movie_set = movies[movies['movieId'].isin(ratings['movieId'])]
# average_rating_per_movie = ratings.groupby('movieId').agg({'rating':'mean'}).reset_index()
# combined = pd.merge(movies, average_rating_per_movie, on='movieId')
# movie_features = combined.pivot_table(index='title', columns='most_common_genre', values='rating')
# movie_features = movie_features.fillna(0)
# matrix_movies_users = csr_matrix(movie_features.values)

# model_KNN = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
# model_KNN.fit(movie_features)

# def recommender_system(movie_name, dataframe, model, number_recommendations):
#     ind = []
#     model.fit(dataframe)
#     idx = process.extractOne(movie_name, combined['title'])[2]
#     # print(f"Movie selected: {df_movies['title'][idx]}, Index selected: {idx}")
#     print('Movie Selected: ', combined['title'][idx], 'Index: ',idx)
#     print('Searching for recomendation....')
#     distances, indices = model.kneighbors(dataframe[idx], n_neighbors=number_recommendations)
#     for i in indices:
#         ind.append(combined['title'][i].where(i!=idx).index)
   
#     selected = pd.Index(ind[0])
#     selected_movie = pd.DataFrame([combined.loc[idx]])
#     selected_movies = combined.loc[selected]
    
#     return (selected_movies, selected_movie)

# # recommendations = recommender_system('shrek', matrix_movies_users,model_KNN, 20)

class RecommendationSystem:
    def __init__(self, title):
        if not title:
            print("Error: Movie title cannot be empty.")
            return
        self.title = title
        self.movies = movies
        self.ratings = ratings
        
    def get_movie(self):
        movie = process.extractOne(self.title, movies['title'])
        return [movie[0], movie[1]]

    def get_movie_Id(self):
        movie = self.get_movie()
        cleaned = movies[movies['title'] == movie[0]]
        return cleaned['movieId'].values[0]
    
    def get_movie_year(self):
        movie = self.get_movie()
        cleaned = movies[movies['title'] == movie[0]]
        return cleaned['year'].values[0]

    def process_movies(self):
        year = self.get_movie_year()
        movie = self.get_movie()
        movie_title = movie[0]
        movie_rows = movies[movies['title'] == movie_title]
        categories = []
        
        unique_genres = []
        for genres in movie_rows['genres'].str.split('|'):
            for genre in genres:
                if genre not in unique_genres:
                    unique_genres.append(genre)

        if len(unique_genres) > 2:
            for index, item in enumerate(unique_genres):
                if index < 2:
                    categories.append(item)
                else:
                    break
        else:
            categories = list(unique_genres)

        dfs = []
        for category in categories:
            df = movies[movies['most_common_genre'] == category]
            dfs.append(df)

        if len(dfs) == 0:
            df = pd.DataFrame(columns=movies.columns)
        else:
            df = pd.concat(dfs)

        df = df[df['year'] >= (year - 10)]
        
        return df
    
    def clean_ratings(self):
        processed = self.process_movies()
        rate = ratings[ratings['movieId'].isin(processed['movieId'])]
        return rate
    
    def ratings_features(self):
        rate = self.clean_ratings()
        ratings_features = rate.pivot(columns='userId', index='movieId', values='rating').fillna(0)
        return ratings_features
        
    def matrix_dataframe(self):
        ratings_features = self.ratings_features()
        matrix_movies_users = csr_matrix(ratings_features.values)
        
        return matrix_movies_users
    
    def recommend(self):
        processed = self.process_movies()
        ratings_features = self.ratings_features()
        title = self.get_movie()
        mat = self.matrix_dataframe()
        movieId = self.get_movie_Id()
        model_KNN = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=22)
        model_KNN.fit(mat)

        row_idx = ratings_features.index.get_loc(movieId)
        
        distances, indices = model_KNN.kneighbors(mat[row_idx, :], n_neighbors=22)
        selected = indices[0]
        selected = selected[selected != row_idx]

        selected_movies = processed.iloc[selected]

        
        return selected_movies
    def searched_movie(self):
        movie = self.get_movie()
        movie_title = movie[0]
        movie_df = movies[movies['title'] == movie_title]
        return movie_df

@app.route('/api/dictionary')
def dictionary():
    word = request.args.get('word')
    # word_movie = movies[movies['title_no_year'] == word]
    
    recomendations = RecommendationSystem(word).recommend()
    searched_movie = RecommendationSystem(word).searched_movie()


    # return jsonify([recommendations[0].to_dict(orient='records'), recommendations[1].to_dict(orient='records')])
    return jsonify([recomendations.to_dict(orient='records'), searched_movie.to_dict(orient='records')])

if __name__ == '__main__':
    app.run(debug=True, port=5100)