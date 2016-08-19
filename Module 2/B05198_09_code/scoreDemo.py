# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 09:16:46 2016

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
X, y = samples_generator.make_classification(n_samples=1000,n_informative=5, n_redundant=0,random_state=42)
le=LabelEncoder()
y=le.fit_transform(y)
Xtrain, Xtest, ytrain, ytest = cross_validation.train_test_split(X, y, test_size=0.5, random_state=1)
clf1=DecisionTreeClassifier(max_depth=2,criterion='gini').fit(Xtrain,ytrain)
clf2= svm.SVC(kernel='linear', probability=True, random_state=0).fit(Xtrain,ytrain)
clf3=LogisticRegression(penalty='l2', C=0.001).fit(Xtrain,ytrain)
pipe1=Pipeline([['sc',StandardScaler()],['mod',clf1]])
mod_labels=['Decision Tree','SVM','Logistic Regression' ]
print('10 fold cross validation: \n')
for mod,label in zip([pipe1,clf2,clf3], mod_labels):
    #print(label)
    auc_scores= cross_val_score(estimator= mod, X=Xtrain, y=ytrain, cv=10, scoring ='roc_auc')
    p_scores= cross_val_score(estimator= mod, X=Xtrain, y=ytrain, cv=10, scoring ='precision_macro')
    r_scores= cross_val_score(estimator= mod, X=Xtrain, y=ytrain, cv=10, scoring ='recall_macro')
    f_scores= cross_val_score(estimator= mod, X=Xtrain, y=ytrain, cv=10, scoring ='f1_macro')
    
    print(label)
    print("auc scores %2f +/- %2f " % (auc_scores.mean(), auc_scores.std()))
    print("precision %2f +/- %2f " % (p_scores.mean(), p_scores.std()))
    print("recall %2f +/- %2f ]" % (r_scores.mean(), r_scores.std()))    
    print("f scores %2f +/- %2f " % (f_scores.mean(), f_scores.std()))
 
