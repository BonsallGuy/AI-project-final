# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 18:28:28 2021

@author: Joshua
"""
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def make_sets(lyrics):
 genres = [
 'Country', 'alt rock', 'Hip Hop']

 LYRIC_LEN = 600
 train = pd.DataFrame()
 test = pd.DataFrame()
 for genre in genres: # loop over each genre
     subset = lyrics[ # create a subset 
         (lyrics.ranker_genre==genre) & 
         (lyrics.lyric.str.len() < LYRIC_LEN)
     ]
     train_set = subset.sample(n=10000, random_state=1, replace = True)
     test_set = subset.drop(train_set.index)
     train = train.append(train_set) # append subsets to the master sets
     test = test.append(test_set)
    
 train = shuffle(train)
 test = shuffle(test)
 return train, test, genres