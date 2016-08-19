# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 13:45:32 2016

@author: dj
"""

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit([[1,2,0], [1, 1, 0], [0, 2, 1], [1, 0, 2]]) 
print(enc.transform([1,2,0]).toarray())
