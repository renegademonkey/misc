
#%% python imports -----------------------
import os
import pandas as pd
import numpy as np
import string
import unicodedata
import itertools
import operator
from pathlib import Path
import __main__ as main

## plotting imports -----------------------
import matplotlib as mpl
mpl.get_backend()
mpl.use('MacOSX')
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style = "whitegrid", font_scale = 1.2)
import pyLDAvis
import pyLDAvis.gensim

## NLP imports -----------------------
from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
import inflect
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud, ImageColorGenerator
from nltk import FreqDist
import gensim


# directories & option settings ---------------------

## set display options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 75)
pd.set_option('display.width', 50000)
pd.set_option('display.max_colwidth', 100)


#%% GET RELATIVE DIRECTORY PATHS

######### ENTER FILE LOCATION HERE
CURRENT_PATH = 'ENTER/CURRENT/SCRIPT/DIRECTORY/HERE/'

# ELSE
if hasattr(main, '__file__'):
    CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))         # only runs in non-interactive mode


DATA_PATH = os.path.join(Path(CURRENT_PATH).parents[0], 'data')
PLOT_PATH = os.path.join(Path(CURRENT_PATH).parents[0], 'plots')

# create directories if needed
if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)
if not os.path.exists(PLOT_PATH):
    os.mkdir(PLOT_PATH)




#%% load data exported from BQ

if CURRENT_PATH != 'ENTER/CURRENT/SCRIPT/DIRECTORY/HERE/':
    feedback_data_raw = pd.read_csv(os.path.join(DATA_PATH, 'miromerge_excl_supporttix-20200818-165724.csv'), encoding = 'utf-8')
else:
    print('PLEASE MANUALLY ENTER FILE PATH LOCATION OF SCRIPT IF RUNNNING IN INTERACTIVE MODE AND RE RUN')

# change dates to timedate format
feedback_data_raw['user_registered_date'] = pd.to_datetime(feedback_data_raw['user_registered_date'])

# explore data
feedback_data_raw.head(200)
feedback_data_raw.dtypes
feedback_data_raw.shape
feedback_data_raw.groupby('user_id').count().shape
feedback_data_raw[feedback_data_raw.user_id.isna()].shape
feedback_data_raw[~feedback_data_raw.user_id.isna()].shape
feedback_data_raw[~feedback_data_raw.user_id.isna()].drop_duplicates(subset = 'user_id').shape
feedback_data_raw[~feedback_data_raw.user_id.isna() & feedback_data_raw.duplicated(subset = 'user_id')].head(100)
feedback_data_raw.groupby('user_id').count().sort_values('user_registered_date', ascending = False).head(50)

# count scores provided
feedback_data_raw[~feedback_data_raw['NPS_score'].isna()].count()
feedback_data_raw[~feedback_data_raw['FF_score'].isna()].count()


# load support ticket data
feedback_support_data_raw = pd.read_csv(os.path.join(DATA_PATH, 'miro_supporttix-20200818-165930.csv'), encoding = 'utf-8')

# explore support ticket data
feedback_support_data_raw.head(100)
feedback_support_data_raw.shape
feedback_support_data_raw.groupby('user_id').count().shape




#%% Ecplore score breakdowns & plot

feedback_data = feedback_data_raw.copy()

# look at numbers of free vs paid and older vs younger users
pd.crosstab(feedback_data['free_vs_paid'], feedback_data['user_age'])
# outputs
# user_age      new_user  old_user
# free_vs_paid
# Free              1590       721
# Paid-for          1262      1267


# get feedback scores by free vs non-freee users
feedback_data.groupby('free_vs_paid')[['FF_score', 'NPS_score']].mean()

# output:
#               FF_score  NPS_score
# free_vs_paid
# Free          3.093190   8.983936
# Paid-for      2.881517   8.735000

# get feedback scores by free vs non-freee users
feedback_data.groupby('user_age')[['FF_score', 'NPS_score']].mean()

# output:
#           FF_score  NPS_score
# user_age
# old_user  3.013158   8.897541
# new_user  2.907945   8.838235


# plot FF by segment, user age
fig, ax = plt.subplots(figsize = (7, 5))
sns.set_palette('Greens')
sns.barplot(x = 'free_vs_paid', y = 'FF_score', hue = 'user_age', data = feedback_data)
plt.ylabel('FF Score'); plt.xlabel('')
plt.title('Feature Feedback score by user segment and age')
plt.legend()
ax.set_ylim([2.5, 3.5])
plt.save(os.path.join(PLOT_PATH, 'ff_by_age_segment.png'))


# plot NPS by segment, user age
fig, ax = plt.subplots(figsize = (7, 5))
sns.set_palette('GnBu')
sns.barplot(x = 'free_vs_paid', y = 'NPS_score', hue = 'user_age', data = feedback_data)
plt.ylabel('NPS Score'); plt.xlabel('')
plt.title('NPS score by user segment and age')
plt.legend()
ax.set_ylim([7, 10])
plt.save(os.path.join(PLOT_PATH, 'nps_by_age_segment.png'))


# get feedback scores by role
role_feedback = feedback_data.groupby('functional_role').agg({'user_id': 'count', 'FF_score': 'mean', 'NPS_score': 'mean'})
# output
#                    user_id  FF_score  NPS_score
# functional_role
# Agile_coach             143  3.242424   9.111111
# Company_management      575  2.948905   9.377358
# Consultant              251  2.886364   9.000000
# Customer_service         47  2.625000   8.000000
# Developer               438  2.988827   8.705882
# Education               412  2.783333   8.458333
# IT                       75  2.705882   8.400000
# Marketing&Sales         195  2.828947   8.352941
# Not detected           1010  2.965278   8.766990
# Operations              254  2.767123   8.466667
# Other                   294  3.112360   8.906250
# Product                 380  2.907975   8.888889
# Project                 348  2.955056   9.230769
# UX/UI Designer          418  3.061321   8.744186

# user counts
print(role_feedback.sort_values('FF_score', ascending = False)[['user_id']].to_string(index = False))
print(role_feedback.sort_values('NPS_score', ascending = False)[['user_id']].to_string(index = False))



# set colors
sns.set_palette('GnBu')


# plot FF Score by role
fig, ax = plt.subplots(figsize = (7, 5))
grp_order = feedback_data.groupby('functional_role')['FF_score'].agg('mean').sort_values().index
sns.barplot(y = 'functional_role', x = 'FF_score', data = feedback_data, order = grp_order[::-1])
ax.set_xlim([2, 4])
plt.gcf().subplots_adjust(left = 0.35)
ax.set_title('Mean FF score by user role')
plt.ylabel('')
plt.save(os.path.join(PLOT_PATH, 'ff_by_role.png'))


# plot NPS scores by role
fig, ax = plt.subplots(figsize = (7, 5))
grp_order = feedback_data.groupby('functional_role')['NPS_score'].agg('mean').sort_values().index
sns.barplot(y = 'functional_role', x = 'NPS_score', data = feedback_data, order = grp_order[::-1])
ax.set_xlim([7, 10])
plt.gcf().subplots_adjust(left = 0.35)
ax.set_title('Mean NPS score by user role')
plt.ylabel('')
plt.save(os.path.join(PLOT_PATH, 'nps_by_role.png'))



#%% prep words

feedback_data.head(50)
feedback_data.dtypes

# get list of english words
english_words = set(nltk.corpus.words.words())
len(english_words)


# do NLP prepping of open text fields (remove html / punctuation /  stopwords, tokenise, lemmatise words)
for col in ['NPS_text', 'FF_text']:
    # fill in blanks
    feedback_data[f'{col}_cleaned'] = feedback_data[col].fillna('')
    # expand contractions
    feedback_data[f'{col}_cleaned'] = feedback_data[f'{col}_cleaned'].apply(lambda x: contractions.fix(x))
    # remove punctuation
    feedback_data[f'{col}_cleaned'] = feedback_data[f'{col}_cleaned'].apply(lambda x: ''.join([c for c in x if c not in string.punctuation]))
    # 'tokenise' words and seperate them
    feedback_data[f'{col}_cleaned'] = feedback_data[f'{col}_cleaned'].apply(lambda x: RegexpTokenizer(r'\w+').tokenize(x))
    # remove stopwords & lowercase all words
    feedback_data[f'{col}_cleaned'] = feedback_data[f'{col}_cleaned'].apply(lambda x: [w.lower() for w in x if w.lower() not in stopwords.words('english')])
    # remove any non-english words (a minority)
    feedback_data[f'{col}_cleaned'] = feedback_data[f'{col}_cleaned'].apply(lambda x: [w for w in x if w in english_words])
    # # remove non unicode characters
    feedback_data[f'{col}_cleaned'] = feedback_data[f'{col}_cleaned'].apply(lambda x: [unicodedata.normalize('NFKD', w).encode('ascii', 'ignore').decode('utf-8', 'ignore') for w in x]) # remove non unicode chars
    # stemm words (not as useful as lemming (see next))
    # feedback_data[f'{col}_cleaned'] = feedback_data[f'{col}_cleaned'].apply(lambda x: ' '.join([SnowballStemmer('english').stem(i) for i in x]))
    # lemmetise words (get common word root)
    feedback_data[f'{col}_cleaned'] = feedback_data[f'{col}_cleaned'].apply(lambda x: [WordNetLemmatizer().lemmatize(w) for w in x if not w.isdigit()])

# preview
feedback_data.head(50)
feedback_data.FF_text.head(50)




#%% wordclouds

# general ff wordcloud
all_ff_text = ' '.join(feedback_data['FF_text_cleaned'].str.join(' '))
words_to_remove = ['grid', 'template']

# FF_wordcloud = WordCloud(stopwords = words_to_remove, max_font_size = 50, max_words = 150, background_color = 'white').generate(all_ff_text)
# plt.figure()
# plt.imshow(FF_wordcloud, interpolation = "bilinear")
# plt.axis("off")
# plt.save(os.path.join(PLOT_PATH, 'ff_by_role.png'))


# positive score ff wordcloud
ff_text_high_score = ' '.join(feedback_data[feedback_data['high_FF'] == 'High']['FF_text_cleaned'].str.join(' '))
words_to_remove = ['grid', 'template', 'table']
# words_to_remove = ['']

FF_wordcloud = WordCloud(stopwords = words_to_remove, max_font_size = 50, max_words = 150, background_color = 'white').generate(ff_text_high_score)
plt.figure(figsize = (10, 7))
plt.imshow(FF_wordcloud, interpolation = "bilinear")
plt.axis("off")
plt.save(os.path.join(PLOT_PATH, 'ff_wordcloud_positive_ff_score_stopwords.png'))


# negative score ff wordcloud
ff_text_low_score = ' '.join(feedback_data[feedback_data['high_FF'] == 'Low']['FF_text_cleaned'].str.join(' '))
words_to_remove = ['grid', 'template', 'table']
# words_to_remove = ['']

FF_wordcloud = WordCloud(stopwords = words_to_remove, max_font_size = 50, max_words = 150, background_color = 'white').generate(ff_text_low_score)
plt.figure(figsize = (10, 7))
plt.imshow(FF_wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.save(os.path.join(PLOT_PATH, 'ff_wordcloud_negative_ff_score_stopwords.png'))



# general nps wordcloud
all_nps_text = ' '.join(feedback_data['NPS_text_cleaned'].str.join(' '))
# words_to_remove = ['grid', 'template']
words_to_remove = ['miro']

FF_wordcloud = WordCloud(stopwords = words_to_remove, max_font_size = 50, max_words = 200, background_color = 'white').generate(all_nps_text)
plt.figure(figsize = (10, 7))
plt.imshow(FF_wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.save(os.path.join(PLOT_PATH, 'nps_wordcloud.png'))


#%% try topic analysis using LDA?


## FF TOPIC MODELLING ------------------

non_informative_words = ['add', 'would', 'could', 'able', 'like', 'use', 'much', 'also', 'miro', 'want', 'love',
                         'one', 'get', 'great', 'need', 'good']

# build dictionary of terms
all_ff_text = ' '.join(feedback_data['FF_text_cleaned'].str.join(' '))
ff_dictionary = gensim.corpora.Dictionary([[w for w in all_ff_text.split() if w not in non_informative_words]])

all_ff_tokenized_text = list(itertools.chain(*feedback_data['FF_text_cleaned'].values.tolist()))
ff_fdist = FreqDist(all_ff_tokenized_text)

# build corpus of bag of words for each comment
ff_corpus = [ff_dictionary.doc2bow(array) for array in feedback_data['FF_text_cleaned'].values.tolist()]

## Re-run with 3 topics ------
#  Latent Dirichlet Allocation (LDA) used to classify text in a document to a particular topic
ff_lda_model =  gensim.models.LdaMulticore(ff_corpus, num_topics = 4, id2word = ff_dictionary,
                                           passes = 50, workers = 4, random_state = 1)

ff_lda_model.show_topics()

# Visualize the topics
ff_vis = pyLDAvis.gensim.prepare(ff_lda_model, ff_corpus, ff_dictionary)
pyLDAvis.show(ff_vis)
pyLDAvis.save_html(ff_vis, os.path.join(PLOT_PATH, 'ff_4_topics.html'))

## Re-run with 3 topics -----
#  Latent Dirichlet Allocation (LDA) used to classify text in a document to a particular topic
ff_lda_model =  gensim.models.LdaMulticore(ff_corpus, num_topics = 3, id2word = ff_dictionary,
                                           passes = 50, workers = 4, random_state = 1)

ff_lda_model.show_topics()

# Visualize the topics
ff_vis = pyLDAvis.gensim.prepare(ff_lda_model, ff_corpus, ff_dictionary)
pyLDAvis.show(ff_vis)
pyLDAvis.save_html(ff_vis, os.path.join(PLOT_PATH, 'ff_3_topics.html'))


# label each ff text as one of the three topics
feedback_data['FF_topics'] = feedback_data['FF_text_cleaned'].apply(lambda x: np.where(len(x) == 0, None, ff_lda_model.get_document_topics(ff_dictionary.doc2bow(x))))
feedback_data['FF_topic'] = feedback_data['FF_topics'].apply(lambda x: max(dict(x).items(), key = operator.itemgetter(1))[0])




## NPS TOPIC MODELLING ------------------

# build dictionary of terms
all_nps_text = ' '.join(feedback_data['NPS_text_cleaned'].str.join(' '))
nps_dictionary = gensim.corpora.Dictionary([[w for w in all_nps_text.split() if w not in non_informative_words]])

# all_nps_tokenized_text = list(itertools.chain(*feedback_data['NPS_text_cleaned'].values.tolist()))
# nps_fdist = FreqDist(all_nps_tokenized_text)

# build corpus of bag of words for each comment
nps_corpus = [nps_dictionary.doc2bow(array) for array in feedback_data['NPS_text_cleaned'].values.tolist()]

## TRY 8 TOPICS -----
#  Latent Dirichlet Allocation (LDA) used to classify text in a document to a nps_dictionary topic
nps_lda_model =  gensim.models.LdaMulticore(nps_corpus, num_topics = 8, id2word = nps_dictionary,
                                           passes = 100, workers = 4, random_state = 1)

nps_lda_model.show_topics()

# Visualize the topics
nps_vis = pyLDAvis.gensim.prepare(nps_lda_model, nps_corpus, nps_dictionary)
pyLDAvis.show(nps_vis)
pyLDAvis.save_html(nps_vis, os.path.join(PLOT_PATH, 'nps_8_topics.html'))

## RE-RUN with 4 TOPICS ----
#  Latent Dirichlet Allocation (LDA) used to classify text in a document to a nps_dictionary topic
nps_lda_model =  gensim.models.LdaMulticore(nps_corpus, num_topics = 4, id2word = nps_dictionary,
                                           passes = 100, workers = 4, random_state = 1)

nps_lda_model.show_topics()

# Visualize the topics
nps_vis = pyLDAvis.gensim.prepare(nps_lda_model, nps_corpus, nps_dictionary)
pyLDAvis.show(nps_vis)
pyLDAvis.save_html(nps_vis, os.path.join(PLOT_PATH, 'nps_4_topics.html'))


# label each NPS text as one of the three topics
feedback_data['NPS_topics'] = feedback_data['NPS_text_cleaned'].apply(lambda x: np.where(len(x) == 0, None, nps_lda_model.get_document_topics(nps_dictionary.doc2bow(x))))
feedback_data['NPS_topic'] = feedback_data['NPS_topics'].apply(lambda x: max(dict(x).items(), key = operator.itemgetter(1))[0])



feedback_data[['NPS_topics', 'NPS_topic']].head(100)
feedback_data.head(10)
feedback_data.dtypes




#%% save munged and modelled data for later use

feedback_data.to_csv(os.path.join(DATA_PATH, 'miromerge_feedback_data_munged.csv'), encoding = 'utf-8')



#%% explore FF topics to see patterns

pd.set_option('display.max_colwidth', 1000)

### FF Topic 0
ff_lda_model.show_topics()[0]
# '0.025*"sticky" + 0.021*"template" + 0.012*"work" + 0.010*"note" + 0.010*"search" 
# + 0.010*"grid" + 0.010*"nice" + 0.008*"easy" + 0.008*"board" + 0.008*"really"'

# sample comments
feedback_data[feedback_data['FF_topic'] == 0]['FF_text'].head(40)
# feature requests: 'split grids', 'combine grids', 'snap to grid', 'highlight search terms', 'join mind-map branches',
# 'typing into cells problematic', 'mindmap nodes difficult to handle', 'easy to add cols & rows', 'split view for wide grids', 'sticky notes'


### FF Topic 1
ff_lda_model.show_topics()[1]
# ' '0.063*"template" + 0.023*"grid" + 0.014*"card" + 0.012*"see" + 0.012*"find" 
# + 0.012*"way" + 0.010*"board" + 0.009*"nice" + 0.009*"color" + 0.009*"make"'

# sample comments
feedback_data[feedback_data['FF_topic'] == 1]['FF_text'].head(40)
# love the template, ideas for improving design: 'make grid transparent', 'checklist', 'more icons + figures', 'auto-space columns',
# 'better colours', 'have column data types', 'rotate grid'


### FF Topic 2
ff_lda_model.show_topics()[2]
#  '0.057*"grid" + 0.036*"text" + 0.032*"table" + 0.028*"cell" + 0.017*"size"
#  + 0.016*"move" + 0.012*"really" + 0.010*"select" + 0.010*"work" + 0.010*"click"'

# sample comments
feedback_data[feedback_data['FF_topic'] == 2]['FF_text'].head(40)
# text & object input/layout: 'vertical align text', 'grid auto-sizing confusing', 'difficult to write', 'smart alignment',
# 'copying text hard', 'font-size control', 'copy/paste not intuitive', 'clicking ot select difficult', 'text resizing', 'easier undo'





#%% explore NPS topics to see patterns

pd.set_option('display.max_colwidth', 1000)

### NPS Topic 0
nps_lda_model.show_topics()[0]
#  '0.020*"better" + 0.019*"table" + 0.017*"tool" + 0.016*"template" + 0.012*"make"
#  + 0.012*"way" + 0.009*"many" + 0.008*"create" + 0.008*"text" + 0.008*"sticky"'

# sample comments
feedback_data[feedback_data['NPS_topic'] == 0]['NPS_text'].head(40)
# layout requests (esp tables): 'table view', 'copying from spreadsheets poor', 'better tables', 'love templates',
# 'more advanced table features', 'better shape alignment'


### NPS Topic 1
nps_lda_model.show_topics()[1]
#  '0.028*"easy" + 0.015*"intuitive" + 0.015*"work" + 0.011*"comfortable" + 0.010*"text" + 0.010*"make"
#  + 0.009*"really" + 0.009*"new" + 0.008*"tool" + 0.008*"working"')

# sample comments
feedback_data[feedback_data['NPS_topic'] == 1]['NPS_text'].head(40)
# Templates: 'more templates, esp design', 'better text formatting', 'layout manager', 'easier sharing', 'breakout boards',


### NPS Topic 2
nps_lda_model.show_topics()[2]
#  '0.018*"find" + 0.014*"make" + 0.012*"board" + 0.010*"card" + 0.010*"time"
#  + 0.010*"way" + 0.009*"design" + 0.009*"team" + 0.008*"currently" + 0.008*"move"'

# sample comments
feedback_data[feedback_data['NPS_topic'] == 2]['NPS_text'].head(40)
# cards & presentation: 'more info on cards', 'hiding stickies', 'better presentation', 'using columns to import stickies,' 'freestyle stickies',
# 'import to presenting software', 'presentation mode', 'integrate Jira cards', 'relationships b/w cards (links, parents etc))', 'more card types'

### NPS Topic 3
nps_lda_model.show_topics()[3]
#  '0.015*"sticky" + 0.011*"tablet" + 0.010*"really" + 0.010*"free" + 0.010*"pen"
#  + 0.010*"note" + 0.010*"time" + 0.009*"board" + 0.008*"still" + 0.008*"product"')

# sample comments
feedback_data[feedback_data['NPS_topic'] == 3]['NPS_text'].head(40)
# comlexity & ease of use: 'confusing - ease of use', 'improved grids,' 'merge grid cells', 'cvs export', 'resizing stickies hard', 'slow U'





#%% Impact of each theme on feedback score

# Here, we can look at the relative impact of each theme to the general feedback score (FF or NPS)
# his is done by calculating overall feedback score (FF, NPS) [A], calculating the FF/NPS score in the subset of responses in OTHER themes [B]
# and then calculating the difference [B - A] to get the relative 'Impact' of that theme on the feedback scores


### FF Theme Impacts
ff_data = feedback_data[feedback_data['FF_topic'].isin([0, 1, 2])][['user_id', 'FF_score', 'FF_topic', 'FF_text']]

ff_mean = ff_data.drop_duplicates(subset = ['user_id', 'FF_text'])['FF_score'].mean()
ff_impact_topic_0 = ff_mean - ff_data[ff_data['FF_topic'].isin([1, 2])].drop_duplicates(subset = ['user_id', 'FF_text'])['FF_score'].mean()
ff_impact_topic_1 = ff_mean - ff_data[ff_data['FF_topic'].isin([0, 2])].drop_duplicates(subset = ['user_id', 'FF_text'])['FF_score'].mean()
ff_impact_topic_2 = ff_mean - ff_data[ff_data['FF_topic'].isin([0, 1])].drop_duplicates(subset = ['user_id', 'FF_text'])['FF_score'].mean()

print(f'Topic 0 Impact: {round(ff_impact_topic_0, 3)}')
print(f'Topic 1 Impact: {round(ff_impact_topic_1, 3)}')
print(f'Topic 2 Impact: {round(ff_impact_topic_2, 3)}')

# output
# Topic 0 Impact: 0.007
# Topic 1 Impact: 0.122
# Topic 2 Impact: -0.13


### NPS Theme Impacts
nps_data = feedback_data[feedback_data['NPS_topic'].isin([0, 1, 2, 3])][['user_id', 'NPS_score', 'NPS_topic', 'NPS_text']]

nps_mean = nps_data.drop_duplicates(subset = ['user_id', 'NPS_text'])['NPS_score'].mean()
nps_impact_topic_0 = nps_mean - nps_data[nps_data['NPS_topic'].isin([1, 2, 3])].drop_duplicates(subset = ['user_id', 'NPS_text'])['NPS_score'].mean()
nps_impact_topic_1 = nps_mean - nps_data[nps_data['NPS_topic'].isin([0, 2, 3])].drop_duplicates(subset = ['user_id', 'NPS_text'])['NPS_score'].mean()
nps_impact_topic_2 = nps_mean - nps_data[nps_data['NPS_topic'].isin([0, 1, 3])].drop_duplicates(subset = ['user_id', 'NPS_text'])['NPS_score'].mean()
nps_impact_topic_3 = nps_mean - nps_data[nps_data['NPS_topic'].isin([0, 1, 2])].drop_duplicates(subset = ['user_id', 'NPS_text'])['NPS_score'].mean()

print(f'Topic 0 Impact: {round(nps_impact_topic_0, 3)}')
print(f'Topic 1 Impact: {round(nps_impact_topic_1, 3)}')
print(f'Topic 2 Impact: {round(nps_impact_topic_2, 3)}')
print(f'Topic 3 Impact: {round(nps_impact_topic_3, 3)}')

# output
# Topic 0 Impact: 0.105
# Topic 1 Impact: 0.052
# Topic 2 Impact: 0.001
# Topic 3 Impact: -0.125

