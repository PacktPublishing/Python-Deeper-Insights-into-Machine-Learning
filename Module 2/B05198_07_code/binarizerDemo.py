# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 11:43:47 2016

@author: dj
"""

from sklearn.preprocessing import Binarizer, Imputer, OneHotEncoder, PolynomialFeatures
from sklearn.decomposition import PCA
import random as rnd
import numpy as np

def binarize(X):
#binarizers at a threshold o 0.5
    bina=Binarizer(5)
   
    print(X)
    print( bina.transform(X))

#binarize(X=[rnd.randint(0,10) for b in range(1,10)])

def onehot():
    enc = OneHotEncoder()
    enc.fit([[1,2,0], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
    print(enc.transform([1,2,0]).toarray())
    
#onehot()
         
def impute():
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    print(imp.fit_transform([[1, 3], [4, np.nan], [5, 6]]))
    
#impute()

def poly():
    X=np.arange(9).reshape(3,3)
    poly=PolynomialFeatures(degree=2)
    print(X)
    print(poly.fit_transform(X))
#poly()


def pca():
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    pca = PCA(n_components=1)
    pca.fit(X)
    print(pca.transform(X))
    
pca()
