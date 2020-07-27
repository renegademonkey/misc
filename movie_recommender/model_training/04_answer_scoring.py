#!/usr/bin/env python3


"""
04_answer_scoring.py

In this file, I load the three pre-trained models form the last analyses
(collaborative filtering model, similarity model (descriptions), & similarity model (metadata))
and use them to answer the questions at hand.

To predict user preference for novel user-movie pairs (using a conventional a 1-5 rating score)
I use a "collaborative filtering" approach, which assumes that
if a film a user likes are also liked by another person, then other films liked by THAT person
are more likely to appeal to the original user

However, this is not perfect, as not every user may have rated enough mutually rated films for the model
learn their preferences or find other users with similar preferences (the 'cold start' problem)

Another way to predict what review a user may give to a film,
is by looking at how that film is rated by all users - a great film is a great film
and irrespective of individual user preference it may get high scores anyway (the wisdom of the crowds).

To answer the second (bonus) question, which was a bit unclear but I assume was asking for ratings of
how similar every movie is to every other movie in the dataset? Based on this assumption I use
the metadata cosine similarity matrix to find similar films as this looked to have better predictive power
than the descriptions. Also, similarity scores are only computed for the first 1000 movies in the list,
as computing for the whole set would take too long without a cluster and result in an oversized file (0-1000 is already a 1/2 GB csv)

NB: The model scoring uses complex models and can thus take over hours to run.
DO NOT try to run this whole file on a local machine, unless you enjoy waiting
(use a computing cluster with over 64GB of RAM)

"""

print('\nSCORING ANSWERS...\n')


#%% import libraries

# basic libraries -----------------------
import pandas as pd
import os
import numpy as np
import pickle
import gc

# plotting imports -----------------------
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style = "whitegrid", font_scale = 1.2)


# ML & data transformation imports ---------------------
from sklearn.preprocessing import MinMaxScaler


# set display options -----------------------
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 75)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', 20)



#%% set fixed path variables


# get current directory & data directory path
# CURRENT_PATH = '/Users/rwlodarski/git_tree/misc_pers/movie_recommender/model_training' # for testing
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA_PATH = os.path.join(CURRENT_PATH, 'training_data')
SAVED_MODEL_PATH = os.path.join(CURRENT_PATH, 'saved_models')
ANSWER_PATH = os.path.join(CURRENT_PATH, 'answers')


# create directories if needed
if not os.path.exists(TRAINING_DATA_PATH):
    os.mkdir(TRAINING_DATA_PATH)
if not os.path.exists(SAVED_MODEL_PATH):
    os.mkdir(SAVED_MODEL_PATH)
if not os.path.exists(ANSWER_PATH):
    os.mkdir(ANSWER_PATH)



#%% load base movie, id, title, mean vote table

print(f'\nLOADING REFERENCE DATA WITH MOVIE_ID, TITLE, WEIGHTED_SCORE INFO...\n')

# reference table for movie id's, titles, & weighted scores
movie_id_title_vote = pd.read_csv(os.path.join(TRAINING_DATA_PATH, 'movie_id_title_vote.csv'), encoding ='utf-8').set_index('Unnamed: 0')
movie_id_title_vote['id'] = movie_id_title_vote['id'].astype('int32')                                           # reduces memory footprint
movie_id_title_vote['vote_weighted_average'] = movie_id_title_vote['vote_weighted_average'].astype('float16')   # reduces memory footprint

# TEST to see if data found# TEST to see if data found
print(f'TESTING TO SEE IF MOVIE DATA FOUND...\n')
length = movie_id_title_vote.shape[0]
if (length > 1):
    print(f'"movie_id_title_vote.csv" data found ({length} rows)\n')
    print(f'PREVIEW:\n{movie_id_title_vote.head(5)}\n')
else:
    print(f'ERROR: "movie_id_title_vote.csv" has no data. \nPlease check folder "{TRAINING_DATA_PATH}" for file\n')

# preview
# movie_id_title_vote.dtypes
# movie_id_title_vote.head()




#%% Optional model accuracy testing

print('\nSKIPPING TEST TO COMPARE GRID SEARCH MODEL VS BASIC MODEL VS USING ACTUAL USER RATINGS,'
      '\nUNCOMMENT THE FOLLOWING SECTION IN THE SCRIPT IF WISH TO RE-RUN TESTS\n')



#%%  testing collaborative filtering model (user's previous ratings) to rate new movies

##########################################
####     UNCOMMENT ALL TO RE-RUN      ####
##########################################
#
# collab_svd_small = pickle.load(open(os.path.join(SAVED_MODEL_PATH, 'surprise_recommender_model_small'), 'rb'))                 # for use on holdout testing dats only
#
# # test set of user-movie ratings (for comparing collab vs other methods)
# ratings_test_data = pd.read_csv(os.path.join(TRAINING_DATA_PATH, 'ratings_test_data.csv'), encoding ='utf-8')
# ratings_test_data.head()
#
# # full set of user-movie ratings
# ratings = pd.read_csv(os.path.join(TRAINING_DATA_PATH, 'ratings.csv'), encoding ='utf-8')
# ratings.head()
#
#
# # answer sheet
# evaluation_ratings = pd.read_csv(os.path.join(TRAINING_DATA_PATH, 'evaluation_ratings.csv'), encoding ='utf-8')
# evaluation_ratings.head()
# evaluation_ratings.shape
# evaluation_ratings.dtypes
#
#
# # TESTING FOR ACCURACY FOR SCORE PREDICTIONS - try different predictions methods on the TEST SET of data only, compare accuracy
#
# # use the small SVD model predict answers on test data
# ratings_test_data['predicted_rating_SVD'] = ratings_test_data.apply(lambda x: collab_svd_small.predict(x['user'], x['item'])[3], axis = 1)
# ratings_test_data.head()
#
#
# # check to see if any of the predictions were 'impossible'
# ratings_test_data['was_impossible'] = ratings_test_data.apply(lambda x: collab_svd_small.predict(x['user'], x['item'])[4]['was_impossible'], axis = 1)
# ratings_test_data.shape
# ratings_test_data.was_impossible.sum()
# # nope
#
#
# # use the actual score a movie received (* 0.5 since original range is 0-10, current range is 0-5)
# # create dictionary of movie:score pairs for fast lookup
# movie_score_dict = movie_id_title_vote.set_index('id')['vote_weighted_average'].to_dict()
# movie_mean_score = np.array(list(movie_score_dict.values())).mean()
#
# # get actual weighted score for each movie, if score not avaialble, use mean value
# ratings_test_data['predicted_rating_actual_score'] = ratings_test_data.apply(lambda x: (movie_score_dict.get(x['item'], movie_mean_score) * 0.5), axis = 1)
#
# # calculate a 'mean' rating of the two methods
# ratings_test_data['predicted_rating_average'] = ratings_test_data[['predicted_rating_SVD', 'predicted_rating_actual_score']].mean(axis = 1)
#
# # preview
# ratings_test_data.head(20)
#
# # calculate accuracies of all the different methods using test data (using RMSE)
# RMSE_SVD_only = ((ratings_test_data['predicted_rating_SVD'] - ratings_test_data['rating']) ** 2).mean() ** 0.5
# RMSE_actual_only = ((ratings_test_data['predicted_rating_actual_score'] - ratings_test_data['rating']) ** 2).mean() ** 0.5
# RMSE_average = ((ratings_test_data['predicted_rating_average'] - ratings_test_data['rating']) ** 2).mean() ** 0.5
#
# """
# RESULTS ==================
# RMSE_SVD_only     : 0.8084661682558013
# RMSE_actual_only  : 1.2270714283818762
# RMSE_average      : 0.9247041607506279
# it looks like the SVD model defaults gives us the best results on the test set,
# so we will use this to make final predictions for the answer sheet
# """


#%% Making final predictions for movie ratings

#%load pre-trained model files from disk

print(f'\nLOADING FINAL COLLABORATIVE FILTERING MODEL...\n')

# movie ratings based on collaborative filtering model
collab_svd = pickle.load(open(os.path.join(SAVED_MODEL_PATH, 'surprise_recommender_model'), 'rb'))   # for making final predictions only


# test predictions with loaded model
# TEST to see if data found
print(f'TESTING TO SEE IF MODEL LOADED...\n')
prediction = collab_svd.predict(uid = 7, iid = 134853)[3]
if (prediction > 0):
    print(f'prediction worked! (score of {prediction})\n')
    print(f'FULL RESULT:\n{collab_svd.predict(uid = 7, iid = 134853)}\n')
else:
    print(f'ERROR: model not working as expected. \nPlease check folder "{TRAINING_DATA_PATH}" for file\n')


# load answer sheet
evaluation_ratings = pd.read_csv(os.path.join(TRAINING_DATA_PATH, 'evaluation_ratings.csv'), encoding ='utf-8')

# TEST to see if data found
print(f'TESTING TO SEE IF EVALUATION_RATINGS DATA FOUND...\n')
length = evaluation_ratings.shape[0]
if (length > 1):
    print(f'data found ({length} rows)\n')
    print(f'PREVIEW:\n{evaluation_ratings.head(5)}\n')
else:
    print(f'ERROR: "evaluation_ratings.csv" has no data. \nPlease check folder "{TRAINING_DATA_PATH}" for file\n')




# make final predictions using the full model (trained on whole dataset) & save

print(f'\nMAKING RATING PREDICTION USING COLLABORATIVE MODEL...\n')

# make predictions for user-movie pairs using model
evaluation_ratings['predicted_rating'] = evaluation_ratings.apply(lambda x: collab_svd.predict(x['userId'], x['movieId'])[3], axis = 1)

print(f'\nPREVIEW OF RATINGS DATA:\n{evaluation_ratings.head(5)}\n')

# save results
print(f'\nSAVING ANSWERS TO {ANSWER_PATH}\n')
evaluation_ratings.to_csv(os.path.join(ANSWER_PATH, 'evaluation_ratings_answers.csv'), encoding = 'utf-8')

print(f'\n=====================================================\n')




#%% to answer the bonus question, movie similarity

"""
We can use previously trained similarity matricies to match each movie to every other movie
since we do not have a standard to compare model accuracy, I am using the descriptive model to make predicitons.
With more time, since both models provided 'reasonable' looking matches,
we could average the similarity ratings of each model to get a final improved similarity rating.
"""


#%% load relevant models needed for rating movie similarity

print(f'\nLOADING DESCRIPTIVE COSINE SIMILARITY MATRIX MODEL...\n')

# movie similarity based on description text
description_cosine_similarity = np.load(os.path.join(SAVED_MODEL_PATH, 'description_cosine_similarity.npy'))
description_cosine_similarity = description_cosine_similarity.astype('float16')  # reduced memory footprint


# testing
print(f'TESTING TO SEE IF MODEL FOUND...\n')
length = description_cosine_similarity.shape[0]
if (length > 1):
    print(f'Model found ({length} rows)\n')
    print(f'PREVIEW:\n{description_cosine_similarity[:10]}\n')
else:
    print(f'ERROR: model has no data. \nPlease check folder "{SAVED_MODEL_PATH}" for file\n')



#%% function to apply similarity score for every OTHER movie for a given movie

print(f'\nWRITING (SIMPLE) FUNCTION TO GET SIMILARITY SDCORES FOR ALL OTHER MOVIES GIVEN ONE MOVIE...\n')

list_of_movie_ids = np.array(movie_id_title_vote[['id']])

### function to get rate movie similarity
def get_movie_similarity_pairs(movie_id):
    # get movie index
    movie_index = movie_id_title_vote[movie_id_title_vote['id'] == movie_id].index.values.astype(int)[0]

    ### get  similarity scores based on cosine matrix
    md_similarity_scores = list(description_cosine_similarity[movie_index].astype('float16'))
    movie_id_similarity_tuples = list(zip([x[0] for x in list_of_movie_ids], md_similarity_scores))

    # clean up
    del(md_similarity_scores)
    gc.collect()

    return movie_id_similarity_tuples



# test the function on one sample
print(f'TESTING TO SEE IF FUNCTION WORKS...\n')
answer = get_movie_similarity_pairs(862)[:10]
length = len(answer)
if (length > 1):
    print(f'Function worked ({length} rows)\n')
    print(f'FOR MOVIE ID 862, FIRST 10 MOVIES AND SIMILARITY SCORES:\n'
          f'{answer}\n')
else:
    print(f'ERROR: function did not work. Plese check code.\n')





#%% Alternative version of similarity function, using average of both similarity models

print(f'\nNOTE: AN IMPROVED MORE COMPLEX FUNCTION COULD BE APPLIED (USING BOTH SIMILARITY MODELS)\n'
      f'HOWEVER THIS IS EXTREMELY MEMORY INTENSIVE AND SHOULD TO BE RUN ON A COMPUTING CLUSTER...\n')

##################################################################
####       UNCOMMENT ALL TO TRY MORE COMPLEX FUNCTION         ####
##################################################################
#
#
# # movie similarity based on description text
# metadata_cosine_similarity = np.load(os.path.join(SAVED_MODEL_PATH, 'metadata_cosine_similarity.npy'))
# metadata_cosine_similarity = metadata_cosine_similarity.astype('float16')
#
#
# def get_movie_similarity_pairs_complex(movie_id):
#
#     # testing
#     movie_id = 110
#
#     # get the index of the movie (same index as the cosine matrix)
#     movie_index = movie_id_title_vote[movie_id_title_vote['id'] == movie_id].index.values.astype(int)[0]
#
#     ### get description similarity scores based on cosine metric, re-scale from 0-1 for comparability
#     desc_similarity = list(enumerate(description_cosine_similarity[movie_index].astype('float16')))
#     desc_similarity_scores = np.array([i[1] for i in desc_similarity]).reshape(-1, 1)
#     desc_similarity_score_scaled = MinMaxScaler((0, 1)).fit_transform(desc_similarity_scores)
#
#     # get metadata similarity scores based on cosine metric, re-scale from 0-1 for comparability
#     md_similarity = list(enumerate(metadata_cosine_similarity[movie_index].astype('float16')))
#     md_similarity_scores = np.array([i[1] for i in md_similarity]).reshape(-1, 1)
#     md_similarity_scores = list(description_cosine_similarity[movie_index].astype('float16'))
#     md_similarity_score_scaled = MinMaxScaler((0, 1)).fit_transform(md_similarity_scores)
#
#     ## get mean movie similarity score based on both models
#     mean_similarity_scores = np.array(desc_similarity_score_scaled + md_similarity_score_scaled) / 2
#
#     # create (movie_id, similarity_core) tuples for each movie
#     movie_id_similarity_tuples = list(zip([x[0] for x in list_of_movie_ids], [x[0] for x in mean_similarity_scores]))
#
#     # clean up
#     del(desc_similarity, desc_similarity_scores, desc_similarity_score_scaled, mean_similarity_scores,
#         md_similarity_score_scaled, md_similarity, md_similarity_scores)
#     gc.collect()
#
#     return movie_id_similarity_tuples
#
#
# # test the function on one sample
# get_movie_similarity_pairs_complex(862)[:10]




#%% applying a simlairty function to every movie in the dataset

print(f'\nSCORING MOVIES SIMILARITIES USING SIMPLE SIMILARITY FUNCTION \n'
      f'(FOR FIRST 1000 MOVIES IN LIST ONLY, DUE TO TIME/MEMORY CONSTRAINTS)\n')

# create base answer table with movie similarity pairs
# NB ONLY SELECTING THE FIRST 1000 movies, otherwise the compute/memory requirements are too onerous
movie_similarities = movie_id_title_vote[['id', 'title']][:1000]


# use map to apply the function element-wise on a series [id] (less memory intensive):
def get_similar(id):
    return get_movie_similarity_pairs(id)
movie_similarities['movie_ids_and_ratings'] = movie_similarities['id'].map(get_similar)

# testing
print(f'TESTING TO SEE ANSWERS...\n')
length = movie_similarities.shape[0]
width = movie_similarities.shape[1]
if ((width == 3) & (length > 1)):
    print(f'Movies scored sucessfully ({length} rows)\n')
    print(f'PREVIEW:\n{movie_similarities[:10]}\n')
else:
    print(f'ERROR: SCORING DID NOT WORK. Please check code\n')

print(f'SAVING ANSWERS TO FOLDER {ANSWER_PATH}\n')
movie_similarities.to_csv(os.path.join(ANSWER_PATH, 'movie_similarity_scores.csv'), encoding = 'utf-8')
