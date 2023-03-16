# Labb for movies recommendations

- My goal in this exercise is to recommend movies to a user based on inputed movie for toy story.

- First i used fuzzywuzzy to return a close match to the movie inputed by the user. This reduces the potential for 
errors since you would need a perfect match for movie you are handling in the dataframe.

- When I get a close enough string as a movie, I reduce the size of my datasets. The original movies dataset is over 58000 rows and the ratings are over 
27 000 000 rows which is obviously too large for my computer to handle. So the best option is to clean these datasets and use what i need.

- I do this by reducing the datsets by at most 2 categories taken from the inputed movie's genres

- I subsequently reduced the ratings to only contain the ratings of the movies found in my cleaned/processed movies dataset

- I create a create pivot daframe using the cleaned ratings dataset

- create a csr_matrix with my pivot dataframe. This helps safe memory usage by only storing non zero values. the csr matrix is then used 
for my recommender algorithm 

#### How KNN works here
- Cosine similarity measures the similarity between two vectors or data points in multidimensional space. It is measured by the cosine of the angle between two vectors or data points. It determines whether these two vectors are pointing in the same direction. It is often used to measure similarity in text analysis.

- When KNN makes inference about a movie, KNN will calculate the “distance” between the target movie and every other movie in its database, then it ranks its distances and returns the top K nearest neighbor movies as the most similar movie recommendations.

## Backend implementation

The Recommendation functionality is implemented using a flask backend

Check out the *app.py* file