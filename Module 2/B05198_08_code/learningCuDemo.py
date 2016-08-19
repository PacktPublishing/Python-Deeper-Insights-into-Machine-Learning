# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 20:04:30 2016

@author: dj
"""
from sklearn.pipeline import Pipeline
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn import datasets

X, y = datasets.make_classification(n_samples=2000,n_informative=2, n_redundant=0,random_state=42)
Xtrain, Xtest, ytrain, ytest = cross_validation.train_test_split(X, y, test_size=0.5, random_state=1)
pipe = Pipeline ([('sc' , StandardScaler()),('clf', LogisticRegression( penalty = 'l2'))])
trainSizes, trainScores, testScores = learning_curve(estimator=pipe, X=Xtrain, y= ytrain,train_sizes=np.linspace(0.1,1,10),cv=10, n_jobs=1)
trainMean=np.mean(trainScores, axis=1)
testMean=np.mean(testScores, axis=1)
testStd=np.std(testScores,axis=1)
plt.plot(trainSizes, trainMean, color='red', marker='o', markersize=5, label = 'training accuracy')
plt.plot(trainSizes, testMean, color='green', marker='s', markersize=5, label = 'validation accuracy')
plt.grid()
plt.xlabel('Number of Training Samples')
plt.ylabel('Accuracy')
plt.legend(loc=0)
plt.show()