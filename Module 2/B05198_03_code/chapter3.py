# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 17:09:23 2016

@author: dj
"""



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def normal(mean = 0, var = 1):    
    sigma = np.sqrt(var)
    x = np.linspace(-3,3,100)
    plt.plot(x,mlab.normpdf(x,mean,sigma))
    plt.show() 

normal(1,0.5)


from scipy.stats import binom
def binomial(x=10,n=10, p=0.5):
    fig, ax = plt.subplots(1, 1)
    x=range(x)
    rv = binom(n, p)
    plt.vlines(x, 0, (rv.pmf(x)), colors='k', linestyles='-')
    
    #plt.legend(loc='best', frameon=False)
    plt.show()
binomial()    


from scipy.stats import binom
def binomial(x=10,n=10, p=0.5):
    fig, ax = plt.subplots(1, 1)
    x=range(x)
    rv = binom(n, p)
    plt.vlines(x, 0, (rv.pmf(x)), colors='k', linestyles='-')
    plt.show()
binomial()

from scipy.stats import poisson
def pois(x=1000):
    xr=range(x)
    ps=poisson(xr)
    plt.plot(ps.pmf(x/2))
pois()

import scipy.stats as stats
def cdf(s1=100,s2=1):  

    x = np.linspace(0,s2 * 100,s1 *2)
    cd = stats.binom.cdf
    plt.plot(x,cd(x, s1, s2))
    plt.show()
cdf()

from PIL import Image
def imageDemo():
    image= np.array(Image.open('data/sampleImage.jpg'))
    plt.imshow(image, interpolation='nearest')
    plt.show()
    print(image.shape)

import soundfile as sf

def audioDemo():
    sig, samplerate = sf.read('data/audioSamp.wav')



    plt.plot(np.abs(np.fft.fft(sig)))
    plt.plot(sig)

audioDemo()

from twitter import Twitter, OAuth

def twit():

    
    
    
    
    apiKey='BykEGOXCp231NJykgrD7G4eU7'
    apiSecret='DJ2iEQBrA8WtAZJ7TDF4HMBVQpSnj6kNX8FPb5F27DSIN2y5Xd'
    
    accesToken='236715091-AlIXFy9gbF8bLbB3Vz7dyYpW0o3bPwissZ8AVbwK'
    secretToken='37rWE9p4wXdmUNkh2TOdCY2BfMWJRWVE5tFvXHzA9xrLP'
    
    #create our twitter object
    t = Twitter(auth=OAuth(accesToken, secretToken,  apiKey, apiSecret))
    #get our home time line
    home=t.statuses.home_timeline()
    #get a public timeline
    anyone= t.statuses.user_timeline(screen_name="abc730")
    #search for a hash tag 
    pycon=t.search.tweets(q="#pycon")
    #
    #pc1=(pycon[0])
    # The screen name of the user who wrote the first 'tweet'
    user=anyone[0]['user']['screen_name']
    #time tweet was created
    created=anyone[0]['created_at']
    #the text of the tweet
    text= anyone[0]['text']
    #user=pycon['search_metadata']['text']

twit()



