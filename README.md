# netflixandchill-knn

## Netflix Recommendation App using Machine Learning ##

EDA - A separate notebook for EDA is provided

Preprocessing:
Data Selection: The model starts by selecting movies from a dataset of Netflix titles (netflix_titles.csv). It focuses specifically on the 'Movie' type and extracts necessary columns (title, listed_in, description).

Data Cleaning: Special characters are removed from the movie descriptions and text is converted to lowercase to standardize the data.

Embedding with FastText:
Sentence Splitting: The cleaned descriptions are split into words to create a corpus suitable for training a FastText embedding model.

FastText Training: A FastText model is trained on the corpus. FastText is a library for efficient learning of word representations and sentence classification. The trained embeddings capture semantic meaning and can handle out-of-vocabulary words through subword information.

Clustering with K-Means:
Feature Vector Transformation: Each movie description is transformed into a feature vector using the trained FastText model.

K-Means Clustering: The model applies K-Means clustering to the vectors, grouping the movies into 26 clusters. This step is used to find movies that are similar to each other in terms of their descriptions.

Recommendation System:
Similarity Search: When a movie title is given to the recommendation system, it finds the cluster that the movie belongs to, computes the similarity between the given movie's description and other movies' descriptions in the same cluster using the FastText model's similarity function, and sorts them by similarity.

Recommendation Output: The system outputs the titles of the top-k (5) most similar movies based on the computed similarity.

Model Saving:
Serialization: The trained K-Means model is saved to disk using joblib for later use, which allows the model to be loaded without retraining.
Usage:
A user can input a movie title to the recommendation_system function, and the system will return a list of recommended movies that are most similar to the input title based on their descriptions.
In summary, the machine learning model combines text preprocessing, FastText embeddings, K-Means clustering, and similarity scoring to recommend movies that are contextually similar to a given movie. The recommendation is based on movie descriptions and grouped by similarity in their content, which should theoretically help users find movies of a similar nature or theme.
