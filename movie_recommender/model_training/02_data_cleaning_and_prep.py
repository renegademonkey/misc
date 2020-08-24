#!/usr/bin/env python3


"""
02_data_cleaning_and_prep.py

In this file the main datafame (metadata) is imported and analysed for variable usefuleness,
data quality, distributions, missing values etc.

Useful variables are  cleaned, transformed, munged, and prepped for modelling

"""

print('\nRUNNING DATA CLEANING & MUNGING...\n')


#%% import libraries

# basic libraries -----------------------
import pandas as pd
import os
import ast
import numpy as np
import datetime

# plotting imports -----------------------
import matplotlib as mpl
# mpl.get_backend()
# mpl.use('MacOSX')
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style = "whitegrid", font_scale = 1.2)


# transformation imports ---------------------
from nltk.stem.snowball import SnowballStemmer


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

# create directories if needed
if not os.path.exists(TRAINING_DATA_PATH):
    os.mkdir(TRAINING_DATA_PATH)


print(f'\nCURRENT WORKING DIRECTORY: {CURRENT_PATH}\n')




#%% IMPORT RAW DATA

# names of data files ot import
data_set_names = ['movies_metadata', 'ratings', 'evaluation_ratings']

print(f'\nIMPORTING {[x+".csv" for x in data_set_names]} DATA FROM DIRECTORY: \n"{TRAINING_DATA_PATH}"\n')

# import all data sets into a dictionary
data_set_dic = {d: pd.read_csv(os.path.join(TRAINING_DATA_PATH,  f'{d}.csv'), encoding ='utf-8', low_memory = False) for d in data_set_names}


# TEST to see if data found
print(f'TESTING TO SEE IF DATA FOUND...\n')
for key, value in data_set_dic.items():
    length = value.shape[0]
    if (length > 1):
        print(f'"{key}.csv" data found ({length} rows)\n')
    else:
        print(f'ERROR: "{key}.csv" has no data. \nPlease check folder "{TRAINING_DATA_PATH}" for file\n')



#%% explore / prep movie metadata

print(f'\nCLEANING METADATA DATABASE\n')

### extract and copy just metadata dataframne -----------------
metadata = data_set_dic['movies_metadata'].copy()
metadata.head(5)
metadata.dtypes


### drop non-useful columns -----------------

print(f'\nPREVIEW OF METADATA:\n {metadata.head(5)}\n')


# drop columns not likely to yield useful predictions (b/c unrelated to movie quality, or already covered by other variables)

# are status, video, adult a useful variable?
# metadata.groupby('status').count()['id']
# metadata.groupby('video').count()['id']
# metadata.groupby('adult').count()['id']

# no, as almost all movies fall into 'Released' category,
# and unreleased/post-production movies unlikely to be useful as recommendations
# and very few video titles
# and very few 'adult' titles

columns_to_drop = ['homepage', 'poster_path', 'imdb_id', 'original_title', 'original_language', 'status', 'video', 'adult']
metadata.drop(columns_to_drop, axis = 1, inplace = True)


print(f'\nDROPPED NON-USEFUL COLUMNS:\n {columns_to_drop}\n')
print(f'\nPREVIEW OF METADATA:\n {metadata.head(5)}\n')





#%% extract relevant information from nested columns -------------------

# names of columns with information stored in dictionaries/json strings
dict_columns = ['genres', 'production_companies', 'production_countries', 'spoken_languages']

print(f'\nEXTRACTING MAIN USEFUL VARIABLES FROM NESTED DICTIONARY/JSON COLUMNS:\n {dict_columns+["collection_list"]}\n')

# convert belongs_to_collection column to dict/json, extract collection ';'name' only and save as new variable, fill NaN's with blanks
metadata['collection_list'] = metadata['belongs_to_collection'].fillna('{}').apply(lambda x: ast.literal_eval(x)).apply(pd.Series)['name'].fillna('')
metadata['collection_list'] = metadata['collection_list'].apply(lambda x: x.split(','))  # turn into array for easier mergeing later on

# extract relevant information form these columns [i.e. 'name'] and store in new columns, create list if multiple items
for col in dict_columns:
     metadata[f'{col}_list'] = metadata[col].fillna('[]').apply(ast.literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

# drop old, extraneous columns
metadata.drop('belongs_to_collection', axis = 1, inplace = True)
metadata.drop(dict_columns, axis = 1, inplace = True)



# metadata.dtypes
print(f'\nPREVIEW OF CLEANED METADATA SO FAR:\n {metadata.head(5)}\n')




#%% deal with missing data -------------------

print(f'\nDEALING WITH MISSING VALUES\n')


# numerical columns

# check data types
# metadata.dtypes

# check missing values
print(f'\nLIST OF MISSING VALUES:\n {metadata.isna().sum()}\n')


# since a few movies are missing id or title (<11), can drop them as not useful
print(f'\nLDROPPING MOVIES WITH NO ID OR TTLE\n')
metadata.dropna(subset = ['id', 'title'], inplace = True)

# check non-numerical columns for weird data
# for col in ['budget', 'popularity', 'id']:
#     print(f'{col}:')
#     metadata[metadata[col].str.len() >= 12][col]


print(f'\nCONVERTING "NUMERICAL" COLUMNS TO NUMBERS\n')

# found some strings mixed in, convert non-numerical to float with coercion
for col in ['budget', 'popularity', 'id']:
    metadata[col] = pd.to_numeric(metadata[col], errors = 'coerce')



# explore numerical columns

df_size = metadata.shape[0]
print(f'\nNEW DATAFRAME SIZE:\n{df_size} rows\n')

# select column names for numerical columns
numerical_cols = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']

# look at distributions of numerical cols
# sns.pairplot(metadata[numerical_cols])

# count values that are missing in numerical columns (nan or 0)
print(f'\nRE-CHECKING MISSING VALUES FOR NUMERICAL COLUMNS')

for col in numerical_cols:
    missing_count = metadata[col].replace({0: np.nan}).isnull().sum()
    print(f'{col} \n Nans: {missing_count}, \n % missing: {round(missing_count/df_size*100, 1)}%\n')


# for 'budget' and 'revenue', there are many missing values or 0's (> 80%)
# as it's unclear why this is, best to change to categorical values with 3 levels: no_value, low, high
# no value = missing or 0, low = 0-50% percentile (of remaining values), high = 50-100% percentile (of remaining values)

cat_conversion_cols = ['budget', 'revenue']

print(f'\nDUE TO MANY MISSING VALUES, CONVERTING: {cat_conversion_cols} to categorical columns (no_, low_, high_)\n')

for col in cat_conversion_cols:
    metadata[f'{col}_cat'] = (pd.cut(
                                    metadata[col],
                                    bins = [0.0, 1.0, metadata[metadata[col] > 1][col].median(), metadata[col].max()],
                                    labels = [f'no_{col}', f'low_{col}', f'high_{col}']
                                ).fillna(f'no_{col}')
                              )
    metadata[f'{col}_cat'] = metadata[f'{col}_cat'].apply(lambda x: x.split(',')) # turn into array for easier mergeing later on
    # metadata[f'{col}_cat'] = metadata[f'{col}_cat'].apply(lambda x: [y for y in x]) # turn into array for easier mergeing later on
    # metadata[f'{col}_cat'] = metadata[f'{col}_cat'].apply(lambda x: ''.join(list(x))) # turn into array for easier mergeing later on
    # print(metadata[f'{col}_cat'].value_counts(normalize = True))
# metadata.head()

# for 'popularity', 'runtime', 'vote_average', 'vote_count', missing values < 7%
# can fill these with the median value (50% percentile)

fill_na_cols = ['popularity', 'runtime', 'vote_average', 'vote_count']
print(f'\nFILLING SPARSE MISSING VALUES FOR {fill_na_cols} WITH MEDIAN VALUE\n')

for col in fill_na_cols:
    metadata[f'{col}_filledna'] = metadata[col].fillna(metadata[col].median())


# for text columns, fill NAs with blank string, so still useable in analyses
fill_na_txt_cols = ['overview', 'tagline', 'collection_list']

for col in fill_na_txt_cols:
    metadata[f'{col}_filledna'] = metadata[col].fillna('')



# drop old variables
# metadata.dtypes
metadata.drop(cat_conversion_cols+fill_na_cols+fill_na_txt_cols, axis = 1, inplace = True)
# metadata.dtypes
# metadata.head(50)





#%% CREATE NEW, MORE USEFUL VARIABLES

### convert release date into 'age' of film

print(f'\nCREATING NEW VARIABLES...\n')

# convery to date
metadata['release_date'] = pd.to_datetime(metadata['release_date'], format='%Y-%m-%d', errors='coerce')

# function to calculate movie age
def movie_age(release_date):
    today = datetime.date.today()
    return today.year - release_date.year - ((today.month, today.day) < (release_date.month, release_date.day))

# calculate movie age and fill missing with median
metadata['movie_age'] = metadata['release_date'].apply(lambda x: movie_age(x))
metadata['movie_age_filledna'] = metadata['movie_age'].fillna(metadata['movie_age'].median())

# drop old columns
metadata.drop(['release_date', 'movie_age'], axis = 1, inplace = True)
metadata = metadata.reset_index()

print(f'\nADDED "movie_age"\n')

# metadata.dtypes
# metadata.head()




#%% NLP PREP

print(f'\nCLEANING & PREPPING DESCRIPTIVE AND METADATA TEXT FOR MODELLING...\n')


### create merged descriptive variable
metadata['full_description'] = metadata['overview_filledna'] + metadata['tagline_filledna']
metadata.drop(['overview_filledna', 'tagline_filledna'], axis = 1, inplace = True)


### also add other key (word) variables to create more robust movie details
metadata['all_text_metadata'] = (metadata['genres_list']
                                     + metadata['production_companies_list']
                                     + metadata['production_countries_list']
                                     + metadata['spoken_languages_list']
                                     + metadata['budget_cat']
                                     + metadata['revenue_cat']
                                     + metadata['collection_list_filledna']
                                 )

# convert into string of words, with underscores going compound words
metadata['all_text_metadata'] = metadata['all_text_metadata'].apply(lambda x: ' '.join([i.replace(' ', '_') for i in x]))

# metadata.head()




### prep words for analysis
# create word stems for easier analysis (ie 'throws' will be treated the same as 'throw')
stem = SnowballStemmer('english')

# remove punctuation/spaces, make lowercase, create word stems using above stemmer, make all lowercase & remove extra blanks
for col in ['full_description', 'all_text_metadata']:
    metadata[f'{col}_punc'] = metadata[col].apply(lambda x: [str.lower(i.replace('[^\w\s]', "")) for i in x.split(' ')])
    metadata[f'{col}_stem'] = metadata[f'{col}_punc'].apply(lambda x: ' '.join([stem.stem(i) for i in x]))
    metadata[f'{col}_stem'] = metadata[f'{col}_stem'].fillna('')


# because some movies rated by only a few people, create weighted vote average for each film
print(f'\nCREATING "weighted_average" review score to account for movies with too many/few reviews...\n')

mean_all_votes = metadata['vote_average_filledna'].mean()
minimum_votes = metadata['vote_count_filledna'].quantile(0.90)

metadata['vote_weighted_average'] = (
                                    (metadata['vote_count_filledna'] / (metadata['vote_count_filledna'] + minimum_votes) * metadata['vote_average_filledna'])
                                    + (minimum_votes / (metadata['vote_count_filledna'] + minimum_votes) * mean_all_votes)
                                    )

metadata_final = metadata[['id', 'title', 'popularity_filledna', 'runtime_filledna', 'vote_weighted_average',
                          'vote_average_filledna', 'vote_count_filledna', 'movie_age_filledna',
                          'full_description_stem', 'all_text_metadata_stem']]


# metadata_final.head()



#%% save munged files for modelling
print(f'\nSAVING MUNGED DATA TO FILE IN "{TRAINING_DATA_PATH}"\n')


# metadata_final.head()
metadata.to_csv(os.path.join(TRAINING_DATA_PATH, 'metadata_tmp.csv'), encoding ='utf-8', index = False)
metadata_final.to_csv(os.path.join(TRAINING_DATA_PATH, 'metadata_final.csv'), encoding ='utf-8', index = False)
# metadata_final = pd.read_csv(os.path.join(TRAINING_DATA_PATH, 'metadata_final.csv'), encoding ='utf-8')

# save small file with just movie id, title & real score
movie_id_title_vote = metadata_final[['id', 'title', 'vote_weighted_average']]
# movie_id_title_vote['id'] = movie_id_title_vote['id'].astype('int32')                                           # reduces memory footprint
# movie_id_title_vote['vote_weighted_average'] = movie_id_title_vote['vote_weighted_average'].astype('float32')   # reduces memory footprint
# movie_id_title_vote.dtypes

movie_id_title_vote.to_csv(os.path.join(TRAINING_DATA_PATH, 'movie_id_title_vote.csv'), encoding ='utf-8')




# read back (for testing)
# metadata = pd.read_csv(os.path.join(TRAINING_DATA_PATH, 'metadata_tmp.csv'), encoding = 'utf-8')
# metadata_final = pd.read_csv(os.path.join(TRAINING_DATA_PATH, 'metadata_final.csv'), encoding = 'utf-8')
# movie_id_title_vote = pd.read_csv(os.path.join(TRAINING_DATA_PATH, 'movie_id_title_vote.csv'), encoding = 'utf-8').set_index('Unnamed: 0')


# movie_id_title_vote.head()
# metadata_final.shape
