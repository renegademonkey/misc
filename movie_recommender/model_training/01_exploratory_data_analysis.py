#!/usr/bin/env python3

"""
01_exploratory_data_analysis.py

In this file the data is imported and explored visually and statistically to examine
variable usefuleness, distributions, data skewness, missing values etc.

Useful variables are  cleaned, transformed, munged, and prepped for modelling

"""

print('\nRUNNING INITIAL DATA EXPLORATION...\n')


#%% import libraries

# basic libraries -----------------------
import pandas as pd
import os
import numpy as np

# plotting imports -----------------------
import matplotlib as mpl
# mpl.get_backend()
# mpl.use('MacOSX')
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style = "whitegrid", font_scale = 1.2)


# set display options -----------------------
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 75)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', 20)



#%% set fixed path variables

# get current directory & data directory path
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA_PATH = os.path.join(CURRENT_PATH, 'training_data')

# create directories if needed
if not os.path.exists(TRAINING_DATA_PATH):
    os.mkdir(TRAINING_DATA_PATH)


#%% import all relevant data

# NB: using low_memory = false due to mixed data types within some columns

# names of data files ot import
data_set_names = ['movies_metadata', 'ratings', 'evaluation_ratings']

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


#%% function: basic preview helper

def preview_data(df, number_rows = 20):
    print(f'DATA SHAPE: \n{df.shape}\n')
    print(f'DATA TYPES: \n{df.dtypes}\n')
    print(f'DATA DESCRIPTION: \n{df.describe()}\n\n')
    print(f'DATA PREVIEW: \n{df.head(number_rows)}\n\n')


#%% preview all imported datasets

for key, value in data_set_dic.items():
    print('\n===================================\n')
    print(f'PREVIEWING "{key}.csv"\n')
    preview_data(value, 5)
    # sns.pairplot(value.select_dtypes(include=[np.number]))

