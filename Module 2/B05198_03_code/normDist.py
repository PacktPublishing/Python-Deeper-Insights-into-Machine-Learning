# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 16:36:54 2015

@author: dj
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import scipy.stats as stats
from scipy.stats import binom
from scipy.stats import poisson

def normal(mean=0,variance=1):
   
    sigma = np.sqrt(variance)
    x = np.linspace(-4,4,100)
    
    plt.plot(x,mlab.normpdf(x,mean,sigma))    
    plt.show()

def cumulative(s1=50,s2=0.2):  

    x = np.linspace(0,s2 * 10,s1 *2)
    cdf = stats.binom.cdf
    plt.plot(x,cdf(x, s1, s2))
    plt.show()
 
def binomial(x=10,n=10, p=0.5):
    fig, ax = plt.subplots(1, 1)
    x=range(x)
    rv = binom(n, p)
    plt.vlines(x, 0, (rv.pmf(x)), colors='k', label= 'label',linestyles='-')
    
    plt.legend(loc='best', frameon=False)
    plt.show()
    
def pois(x=1000):
    xr=range(x)
    ps=poisson(xr)
    plt.plot(ps.pmf(x/2))
    
   # k= np.random()
    #plt.plot(poisson.pmf(k,.1)) 
    
    