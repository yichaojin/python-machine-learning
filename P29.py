# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 14:52:21 2018

@author: hasee
"""
import numpy as np
#import sys
#sys.path.append(r'E:\研究生\算法书籍\python machine learning')
from Perceptron import Perceptron
import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.tail()
import matplotlib.pyplot as plt
plt.subplot(3,1,1)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values
plt.scatter(X[:50, 0], X[:50, 1],color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],color='blue', marker='x', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()
plt.subplot(3,1,2)
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()

#from matplotlib.colors import ListedColormap
from plot_decision_regions import plot_decision_regions 

plt.subplot(3,1,3)        
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal lenght [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()        
        