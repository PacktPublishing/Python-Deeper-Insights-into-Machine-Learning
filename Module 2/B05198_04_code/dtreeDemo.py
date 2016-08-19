# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 11:10:56 2015

@author: dj
"""


from sklearn import tree
import os

names=['size','scale','fruit','butt']
labels=[1,1,1,1,0,0,0,0]
p1=[2,1,0,1]
p2=[1,1,0,1]
p3=[1,1,0,0]
p4=[1,1,0,0]
n1=[0,0,0,0]
n2=[1,0,0,0]
n3=[0,0,1,0]
n4=[1,1,0,0]
data=[p1,p2,p3,p4,n1,n2,n3,n4]
def pred(test, data=data):
    dtre=tree.DecisionTreeClassifier()
    dtre=dtre.fit(data,labels)
    print(dtre.predict([test]))
    with open('conifer2.dot', 'w') as f:
        f=tree.export_graphviz(dtre,out_file=f)
    os.unlink('conifer2.dot')
        

pred([0,1,0,1]) 
#pred([2,0,1,1]) 
 