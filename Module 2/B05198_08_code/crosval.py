# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 18:10:44 2016

@author: dj
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
    






from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import samples_generator
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score 
from sklearn.pipeline import Pipeline
X, y = samples_generator.make_classification(n_informative=5, n_redundant=0,random_state=42)
le=LabelEncoder()
y=le.fit_transform(y)
Xtrain, Xtest, ytrain, ytest = cross_validation.train_test_split(X, y, test_size=0.5, random_state=1)
clf1=DecisionTreeClassifier(max_depth=2,criterion='gini').fit(Xtrain,ytrain)
clf2= svm.SVC(kernel='linear', probability=True, random_state=0).fit(Xtrain,ytrain)
clf3=LogisticRegression(penalty='l2', C=0.001).fit(Xtrain,ytrain)
pipe1=Pipeline([['sc',StandardScaler()],['mod',clf1]])
pipe2=Pipeline([['sc',StandardScaler()],['mod',clf3]])
mod_labels=['Decision Tree','SVM','Logistic Regression' ]
print('10 fold cross validation: \n')
for mod,label in zip([pipe1,clf2,pipe2], mod_labels):
    #print(label)
    scores= cross_val_score(estimator= mod, X=Xtrain, y=ytrain, cv=10, scoring ='roc_auc')
    print("scores %2f +/- %2f [%s]" % (scores.mean(), scores.std(), label))





#pipe1= make_pipeline(preprocessing.StandardScaler())
#cross_validation.cross_val_score(pipe1)

#from sklearn.pipeline import make_pipeline
#clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
#cross_validation.cross_val_score(clf, iris.data, iris.target)
#scaler = preprocessing.StandardScaler().fit(Xtrain)
#Xtrain_trans = scaler.transform(Xtrain)
#mod1 = svm.SVC(C=1).fit(Xtrain_trans, ytrain)
#Xtest_trans = scaler.transform(Xtest)
#print(mod1.score(Xtest_trans, ytest))  



