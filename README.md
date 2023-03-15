# Labb for movies recommendations

## 1.3a How my system works

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
- From the csr matrix, each row is a vector. Each of these vectors is a movie since I had my movieIds as rows in the pivot dataframe. These vectors are found in this high 
dimensional matrix space. The KNN recommendation here works by checking the angle between my inputed movie and returning the K nearest other vectors. The other vector vectors in this matrix with smaller angles compared to my movie will be returned 

<img src="../assets/cos.webp" alt="description of the image" width="300" height="200">

A good example is the image above. Joao Felix and Messi are similar, but Jaoa has fewer years of play and doesnt have as many ratings but is  very similar to messi as opposed
to Cristiano who is quite different but simailar in amout of ratings as Messi. A euclidean distance would have picked Messi and Ronaldo where a cosine would pick Joao as similar to Messi.