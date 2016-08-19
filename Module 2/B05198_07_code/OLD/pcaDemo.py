# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 09:26:25 2016

@author: dj
"""

import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=1)
pca.fit(X)
xt=pca.transform(X)