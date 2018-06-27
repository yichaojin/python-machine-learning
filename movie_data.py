# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 19:08:08 2018

@author: hasee
"""

import pyprind
import pandas as pd
import os
pbar = pyprind.ProgBar(50000)
labels = {'pos':1, 'neg':0}
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
            path ='E:/网页下载/aclImdb_v1/aclImdb/%s/%s' % (s, l)
            for file in os.listdir(path):
                with open(os.path.join(path, file), 'rb') as infile:
                    txt = infile.read()
                df = df.append([[txt, labels[l]]], ignore_index=True)
                pbar.update()
df.columns = ['review', 'sentiment']

import numpy as np
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('E:/网页下载/aclImdb_v1/aclImdb/movie_data.csv', index=False)