#!/usr/bin/env python3


"""
03_model_training.py

Here we start to build predictive models to be used to answer the task questions

We start with a collaborative filtering model, based on previous user scores

For movie comparing similarity, I use parts of the movie metadata
(text vars only due to time constraints) to create models predicting which films
are most similar to any other given film
In future this be a useful way to recommend a user has liked in the past
where collaborative filtering is not too accurate

In future, these models could be combined into an ensamble model for greater accuracy

====================================================

NB: The model training parts of this project take up to an hour to train each model
or do matrix multiplication. Furthermore, each model /output can be many GB in size.
This improves accuracy (the goal here), at the expense of complexity/latency
DO NOT try to run this whole file on a local machine, unless you enjoy waiting
(use a computing cluster with over 64GB of RAM)

====================================================

"""

print('\nTRAINING MODELS...\n')
print('\nWARNING: This may take several hours, depending on your system setup...\n')



#%% import libraries

# basic libraries -----------------------
import pandas as pd
import os
import numpy as np
import pickle

# plotting imports -----------------------
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style = "whitegrid", font_scale = 1.2)


# ML & data transformation imports ---------------------
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import surprise


# set display options -----------------------
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 75)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', 20)



#%% set fixed path variables

# get current directory & data directory path
# CURRENT_PATH = '/Users/rwlodarski/git_tree/misc_pers/movie_recommender/model_training'  # for testing
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA_PATH = os.path.join(CURRENT_PATH, 'training_data')
SAVED_MODEL_PATH = os.path.join(CURRENT_PATH, 'saved_models')

# create directories if needed
if not os.path.exists(TRAINING_DATA_PATH):
    os.mkdir(TRAINING_DATA_PATH)
if not os.path.exists(SAVED_MODEL_PATH):
    os.mkdir(SAVED_MODEL_PATH)




#%% Optional parameters search and testing

print('\nPREVIOUS GRID "PARAMETER GRID SEARCH" DID NOT IMPROVE RESULTS'
      '\nUSING MODEL DEFAULT PARAMETERS'
      '\n\n(It is possible to re-run grid search on new data by uncommenting the relvant code in this script and re-running)\n')


#%%  testing collaborative filtering model (user's previous ratings) to rate new movies

##########################################
####     UNCOMMENT ALL TO RE-RUN      ####
##########################################

# # here using the 'surprise' library to do collaborative filtering on the ratings of users
#
# # import ratings data
# ratings = pd.read_csv(os.path.join(TRAINING_DATA_PATH, 'ratings.csv'), encoding ='utf-8')
#
# # preview
# ratings.describe()
# ratings.dtypes
# ratings.shape
#
# # prep data for modelling
# ratings  = ratings[['userId', 'movieId', 'rating']]
# ratings.columns = ['user', 'item', 'rating']
#
#
# # also create seperate data from training & testing (for use when comparing against other models)
# ratings = ratings.copy()
# ratings_train_data = ratings.sample(frac = 0.80, random_state = 0)
# ratings_test_data = ratings.drop(ratings_train_data.index)
# ratings_tuning_data = ratings_train_data.sample(frac = 0.20, random_state = 0)
#
# # save the test data for alter so can compare prediction methods
# ratings_test_data.to_csv(os.path.join(TRAINING_DATA_PATH, 'ratings_test_data.csv'), encoding ='utf-8')
#
#
# # convert data df into 'surprise' format data
# reader = surprise.Reader(rating_scale = (1, 5))
#
# ratings_data_all_sp = surprise.Dataset.load_from_df(ratings, reader = reader)
# ratings_train_data_sp = surprise.Dataset.load_from_df(ratings_train_data, reader = reader)
# ratings_test_data_sp = surprise.Dataset.load_from_df(ratings_test_data, reader = reader)
# ratings_tuning_data_sp = surprise.Dataset.load_from_df(ratings_tuning_data, reader = reader)




### model paramater tuning (grid search - UNCOMMENT TO RE-RUN - TAKES A VERY LONG TIME)

# NB, before full model training we can do a basic search to see which parameters
# give the best result. we will use a small subset of the data
# to find the best set of parameters to use in later models

### try different paramaters to SCD defaults to see if accuracy is improved
# param_grid = {'n_factors': [50, 150], 'n_epochs': [10, 30], 'lr_all': [0.005, 0.02], 'reg_all': [0.2, 0.6]}
#
# # do grid search
# grid_search = surprise.model_selection.GridSearchCV(surprise.SVD, param_grid, measures = ['rmse', 'mse'], cv = 3)
#
# # train model using grid
# grid_search.fit(ratings_tuning_data_sp)
#
# # get the model with the RMSE score
# print(grid_search.best_score['rmse'])
#
# # get combination of parameters that gave the best RMSE score
# print(grid_search.best_params['rmse'])
# print(grid_search.best_estimator)
# print(grid_search.cv_results)
#
# best grif parameters:
# {'n_factors': 50, 'n_epochs': 30, 'lr_all': 0.005, 'reg_all': 0.2}


### test/train model model training

# build a training set for surprise to use ALL data as training data (with cross-validation for testing)
# ratings_train_data_trainset = ratings_train_data_sp.build_full_trainset()

# Singular Value Decomposition (SVD)
# svd_train = surprise.SVD()                      # default parameters better than grid search results - using defaults
# svd_train_tuned = surprise.SVD(n_factors = 50, n_epochs = 30, lr_all = 0.005, reg_all = 0.2)


# Train the algorithm on the train_set
# svd_train.fit(ratings_train_data_trainset)
# svd_train_tuned.fit(ratings_train_data_trainset)




### test on holdout (test) set

# make predictions on unseen test data
# ratings_test_data['predicted_rating'] = ratings_test_data.apply(lambda x: svd_train.predict(x['user'], x['item'])[3], axis = 1)
# ratings_test_data['predicted_rating'] = ratings_test_data.apply(lambda x: svd_train_tuned.predict(x['user'], x['item'])[3], axis = 1)
# ratings_test_data['predicted_rating'] = ratings_test_data.apply(lambda x: svd_train_tuned.predict(x['user'], x['item'])[3], axis = 1)
# ratings_test_data.head()

# calculate accuracy score on test data (using RMSE)
# RMSE = ((ratings_test_data['predicted_rating'] - ratings_test_data['rating']) ** 2).mean() ** 0.5

# The RMSE of the collaborative model (only) is AT MOST:               RMSE = 0.81
# The RMSE of the collaborative model with tuning is AT MOST:          RMSE = 0.89
# It appears that tuning does not improve the results
# [this can happen when focusing on only one method af assessment, ie RMSE]
# The final model is likely to be slightly more accurate as it will be trained
# on the whole data set (with cross-validation on holdout test sets)

# save the trained recommender model for future use
# pickle.dump(svd_train, open(os.path.join(SAVED_MODEL_PATH, 'surprise_recommender_model_small'), 'wb'))
# pickle.dump(svd_train_tuned, open(os.path.join(SAVED_MODEL_PATH, 'surprise_recommender_model_small_tuned'), 'wb'))

# load the model from disk
# loaded_svd_small = pickle.load(open(os.path.join(SAVED_MODEL_PATH, 'surprise_recommender_model_small_tuned'), 'rb'))







#%% full model training

print(f'\nTRAINING COLLABORATIVE FILTERING MODEL USING THE "SURPRISE" ML LIBRARY\n')

# import ratings data
ratings = pd.read_csv(os.path.join(TRAINING_DATA_PATH, 'ratings.csv'), encoding ='utf-8')


# TEST to see if data found
print(f'TESTING TO SEE IF RATINGS DATA FOUND...\n')
length = ratings.shape[0]
if (length > 1):
    print(f'"ratings.csv" data found ({length} rows)\n')
    print(f'PREVIEW:\n{ratings.head(5)}\n')
else:
    print(f'ERROR: "ratings.csv" has no data. \nPlease check folder "{TRAINING_DATA_PATH}" for file\n')


# preview
# ratings.describe()
# ratings.dtypes
# ratings.shape


# prep data for modelling
ratings = ratings[['userId', 'movieId', 'rating']]
ratings.columns = ['user', 'item', 'rating']

# convert data df into 'surprise' format data
print(f'\nCONVERTING DATA TO "SURPRISE" LIBRARY DATA FORMAT...\n')
reader = surprise.Reader(rating_scale = (1, 5))
ratings_data_all_sp = surprise.Dataset.load_from_df(ratings, reader = reader)


# build a training set for surprise to use ALL data as training data (with cross-validation for testing)
print(f'\nCREATING TRAINING DATASET...\n')
train_data = ratings_data_all_sp.build_full_trainset()

# load SVD algorithm.
svd_all = surprise.SVD()
# svd_all_tuned = surprise.SVD(n_factors = 50, n_epochs = 30, lr_all = 0.005, reg_all = 0.2)

# Train the algorithm on the train_set
print(f'\nTRAINING MODEL (this can take a long time)...\n')
svd_all.fit(train_data)

print(f'\nTRAINING COMPLETE!, SAVING MODEL TO DISK IN FOLDER\n"{SAVED_MODEL_PATH}"\n')

# save the trained recommender model for future use
pickle.dump(svd_all, open(os.path.join(SAVED_MODEL_PATH, 'surprise_recommender_model'), 'wb'))
# pickle.dump(svd_all_tuned, open(os.path.join(SAVED_MODEL_PATH, 'surprise_recommender_model_tuned'), 'wb'))

# load the model from disk
# svd_reloaded = pickle.load(open(os.path.join(SAVED_MODEL_PATH, 'surprise_recommender_model'), 'rb'))

# test predictions with loaded model
# svd_all.predict(uid = 1, iid = 100)
# svd_reloaded.predict(uid = 1, iid = 100)


print(f'\n=================================================================================\n')




#%% use movie meta data & descriptions to find similar movies

print(f'\nREADING IN PRE-PREPPED METADATA...\n')

### read_prepped metadata file from disk
print(f'\nREADING IN PRE-PREPPED METADATA FROM "{TRAINING_DATA_PATH}"\n')
metadata_final = pd.read_csv(os.path.join(TRAINING_DATA_PATH, 'metadata_final.csv'), encoding ='utf-8')

# check for NaNs and re-fill with blanks ('')
# metadata_final.isna().sum()
# metadata_final['full_description_stem'] = metadata_final['full_description_stem'].fillna('')

# TEST
print(f'\nTESTING TO SEE IF METADATA FOUND...\n')
length = metadata_final.shape[0]
if (length > 1):
    print(f'\nMETADATA data found ({length} rows)')
    print(f'\nDATA PREVIEW: \n{metadata_final.head()}\n')
else:
    print(f'\nERROR: METADATA FILE has no data. \nPlease check folder "{TRAINING_DATA_PATH}" for file\n')

# metadata_final.head()
# metadata_final.vote_weighted_average.describe()





#%% find movie similarity score based on full description text only

print('\nTRAINING COSINE SIMILARITY MODEL BASED ON DESCRIPTIVE + TAGLINE TEXT...\n')


# create on long string to describe each movie
# metadata_final['all_words_stem'] = metadata_final['full_description_stem'] + metadata_final['all_text_metadata_stem']
# metadata_final.head()

# vectorise description words based on usage in the document (inverse frequency - TF-IDF)
tdif = TfidfVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0, stop_words = ['english'])
tdif_matrix = tdif.fit_transform(metadata_final['full_description_stem'])

# tdif_matrix.shape

# to calculate similarity between movie descriptions, calculate cosine similarity (dot matrix multiplication using linear_kernel)
# description_cosine_similarity = cosine_similarity(tf_matrix, tf_matrix)
description_cosine_similarity = linear_kernel(tdif_matrix, tdif_matrix)

# TEST
print(f'\nTESTING MODEL PREVIEW...\n')
length = len(description_cosine_similarity)
if (length > 1):
    print(f'\nMODEL data found ({length} rows)')
    print(f'\nMODEL PREVIEW: \n{description_cosine_similarity[:5]}\n')
else:
    print(f'\nERROR: MODEL has no data. \nPlease check code.\n')

print(f'\nSAVING DESCRIPTIVE MODEL TO "{SAVED_MODEL_PATH}"')

### save model to file (convert to int8 to save on memory)
np.save(os.path.join(SAVED_MODEL_PATH, 'description_cosine_similarity'), description_cosine_similarity.astype('floar32'))
# description_cosine_similarity = np.load(os.path.join(SAVED_MODEL_PATH, 'description_cosine_similarity.npy'))




#%% find movie similarity score based on metadata text only

print('\nTRAINING COSINE SIMILARITY MODEL BASED ON METADATA TEXT...\n')

# vectorise description words based on usage in the document (NOR inverse frequency, since many relevant tags will be very common)
cv = CountVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0, stop_words = ['english'])
cv_matrix = cv.fit_transform(metadata_final['all_text_metadata_stem'])

# to calculate similarity between movie metadata, calculate cosine similarity (dot matrix multiplication using linear_kernel)
metadata_cosine_similarity = linear_kernel(cv_matrix, cv_matrix)

# TEST
print(f'\nTESTING MODEL PREVIEW...\n')
length = len(metadata_cosine_similarity)
if (length > 1):
    print(f'\nMODEL data found ({length} rows)')
    print(f'\nMODEL PREVIEW: \n{metadata_cosine_similarity[:5]}\n')
else:
    print(f'\nERROR: MODEL has no data. \nPlease check code.\n')


print(f'\nSAVING DESCRIPTIVE MODEL TO "{SAVED_MODEL_PATH}"')

### save model to file (convert to int8 to save on memory)
np.save(os.path.join(SAVED_MODEL_PATH, 'metadata_cosine_similarity'), metadata_cosine_similarity.astype('float32'))
# metadata_cosine_similarity = np.load(os.path.join(SAVED_MODEL_PATH, 'metadata_cosine_similarity.npy'))





#%% check predictions made by two methods against known movies to see which seems more accurate

# get list of movie ids, titles, ratings and indicies of movies
movie_id_title_vote = pd.read_csv(os.path.join(TRAINING_DATA_PATH, 'movie_id_title_vote.csv'), encoding ='utf-8').set_index('Unnamed: 0')

### function to get most similar movies
def get_similar_movie(movie_id, cosine_matrix_to_use, number_of_predictions = 10):

    # testing
    # movie_id = 100; number_of_predictions = 10
    # cosine_matrix_to_use = description_cosine_similarity

    # get the index of the movie (same index as the cosine matrix)
    movie_index = movie_ids[movie_ids['id'] == movie_id].index.values.astype(int)[0]

    # get similarity scores based on cosine metric passed, re-order and re-scale from 1-5 (rating score)
    similarity_scores = list(enumerate(cosine_matrix_to_use[movie_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:number_of_predictions]
    similarity_score = np.array([i[1] for i in similarity_scores]).reshape(-1, 1)
    similarity_score_scaled = MinMaxScaler((0, 1)).fit_transform(similarity_score)

    # add movie similarity score to dataframe with movie id, title, similarity score
    movie_indexes = [i[0] for i in similarity_scores]
    list_of_similar_movies = movie_ids.iloc[movie_indexes].reset_index()
    list_of_similar_movies['similarity_score'] = similarity_score_scaled
    return list_of_similar_movies


print(f'\nRUNNING SANITY CHECK - TESTING MOST SIMILAR MOVIES BASED ON TWO MODELS')


# test on a sample of popular movies
movie_names = ['Armageddon', 'Jaws', 'Star Wars', 'The Godfather']

movie_ids = metadata_final[['id', 'title', 'vote_weighted_average']]

for movie in movie_names:
    # get movie id
    movie_id = movie_ids[movie_ids['title'] == movie].id.values.astype(int)[0]

    # get predictions
    print(f'\n\nMovies similar to "{movie}":\n')

    print(f'Based on description similarity:\n {get_similar_movie(movie_id, description_cosine_similarity, 10)}\n')
    print(f'Based on metadata similarity:\n {get_similar_movie(movie_id, metadata_cosine_similarity, 10)}\n')




#%% result

print('\nIt appears that the "descriptive" similarity method is more accurate. \n'
      'Theoretically, the model accuracy could be improved by combining ratings \n'
      'from these two models, and eventually including other numerical metadata variables.')
