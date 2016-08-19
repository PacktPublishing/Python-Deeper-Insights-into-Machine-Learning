# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:04:52 2015

@author: dj
"""

import numpy as np
from scipy.optimize import minimize
def rosen(x):
	return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def griewank(x):
    gr=[]
    for i ,xi in enumerate(x):         
        gr.append(1.00025 * sum(x*x) -np.prod(np.cos(xi/(i**.5))))
    return(gr)
def cube(x):
    return x**3    

def nMin(funct,x0):

    return(minimize(rosen, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True}))

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
x1 = np.array([0.1,0.5, 0.8, 1, 1.2])
nMin(rosen,x0)
nMin(griewank,x1)
nMin(cube,x0)
