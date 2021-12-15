# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 21:33:42 2021

@author: Joshua
"""

import pandas as pd
import numpy as np
import nltk

def make_data():
 df=pd.read_csv('lyrics1.csv')
 df.append(pd.read_csv('lyrics2.csv'))

 df['ranker_genre']=np.where(
    (df['ranker_genre'] == 'screamo')|
 (df['ranker_genre']== 'punk rock')|
 (df['ranker_genre'] == 'heavy metal'),
 'alt rock',df['ranker_genre'])

 group = ['song', 'year', 'album', 'genre', 'artist', 'ranker_genre']
 l_b_s = df.sort_values(group)\
         .groupby(group).lyric.apply(' '.join).apply(lambda x: x.lower()).reset_index(name='lyric')
 
 l_b_s["lyric"] = l_b_s['lyric'].str.replace(r'[^\w\s]','')#replace num-alpha numeric characters
 return l_b_s
def prepare_song():
 song = open('input.txt', 'rt').read().lower()
 for line in song:
     song = song.replace(r'[^\w\s]','')
 return song
