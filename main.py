# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 21:33:42 2021

@author: Joshua
"""

import pandas as pd
import numpy as np
import nltk
from sklearn.utils import shuffle

from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
from make_data import make_data
from make_data import prepare_song 
from make_sets import make_sets
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns
lyrics = make_data()
train,test, genres = make_sets(lyrics)

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

stop = list(set(stopwords.words('english'))) # stopwords
wnl = WordNetLemmatizer() # lemmatizer

def tokenizer(x): # custom tokenizer
    return (
        wnl.lemmatize(w) 
        for w in word_tokenize(x)
        if len(w) == 2 and w.isalnum() 
    )                                 

# define our model
text_clf = Pipeline(
    [('vect', TfidfVectorizer(
        ngram_range=(1, 2),
        tokenizer=tokenizer,
        stop_words=stop,
        max_df=0.3,
        min_df=4)),
        ('tfidf', TfidfTransformer()),
     ('clf', MultinomialNB(alpha=0.1))])

# train model
text_clf.fit(train.lyric, train.ranker_genre)  

# score model
predicted = text_clf.predict(test.lyric)
print(np.mean(predicted == test.ranker_genre))
print(text_clf.predict([prepare_song()]))
mat = confusion_matrix(test.ranker_genre, predicted)
sns.heatmap(
    mat.T, square=True, annot=True, fmt='d', cbar=True,
    xticklabels= genres, 
    yticklabels=genres
)
plt.xlabel('correct')
plt.ylabel('predicted')


precision, recall, fscore, support = precision_recall_fscore_support(test.ranker_genre, predicted)

for n,genre in enumerate(genres):
    genre = genre.upper()
    print(genre+'_precision: {}'.format(precision[n]))
    print(genre+'_recall: {}'.format(recall[n]))


