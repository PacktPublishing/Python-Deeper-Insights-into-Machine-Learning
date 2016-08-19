# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 10:22:01 2016

@author: dj
"""



#import numpy as np
#from math import sqrt 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)  
 
import pandas as pd 
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

userRatings={'Dave': {'Dark Side of Moon': 9.0,
  'Hard Road': 6.5,'Symphony 5': 8.0,'Blood Cells': 4.0},'Jen': {'Hard Road': 7.0,'Symphony 5': 4.5,'Abbey Road':8.5,'Ziggy Stardust': 9,'Best Of Miles':7},'Roy': {'Dark Side of Moon': 7.0,'Hard Road': 3.5,'Blood Cells': 4,'Vitalogy': 6.0,'Ziggy Stardust': 8,'Legend': 7.0,'Abbey Road': 4},'Rob': {'Mass in B minor': 10,'Symphony 5': 9.5,'Blood Cells': 3.5,'Ziggy Stardust': 8,'Black Star': 9.5,'Abbey Road': 7.5},'Sam': {'Hard Road': 8.5,'Vitalogy': 5.0,'Legend': 8.0,'Ziggy Stardust': 9.5,'U2 Live': 7.5,'Legend': 9.0,'Abbey Road': 2},'Tom': {'Symphony 5': 4,'U2 Live': 7.5,'Vitalogy': 7.0,'Abbey Road': 4.5},'Kate': {'Horses': 8.0,'Symphony 5': 6.5,'Ziggy Stardust': 8.5,'Hard Road': 6.0,'Legend': 8.0,'Blood Cells': 9,'Abbey Road': 6}}

# Returns a distance-based similarity score for user1 and user2
def distance(prefs,user1,user2):
    # Get the list of shared_items
    si={}
    for item in prefs[user1]:
        if item in prefs[user2]:
            si[item]=1            
    # if they have no ratings in common, return 0
    if len(si)==0: return 0    
    # Add up the squares of all the differences
    sum_of_squares=sum([pow(prefs[user1][item]-prefs[user2][item],2)
    for item in prefs[user1] if item in prefs[user2]])    
    return 1/(1+sum_of_squares)    

def Matches(prefs,person,n=5,similarity=pearsonr):
    scores=[(similarity(prefs,person,other),other)
        for other in prefs if other!=person]    
    scores.sort( )
    scores.reverse( )
    return scores[0:n]

def getRecommendations(prefs,person,similarity=pearsonr):
    totals={}
    simSums={}
    for other in prefs:       
        if other==person: continue
        sim=similarity(prefs,person,other)
        if sim<=0: continue
        for item in prefs[other]:            
            # only score albums not yet rated
            if item not in prefs[person] or prefs[person][item]==0:
                # Similarity * Score
                totals.setdefault(item,0)
                totals[item]+=prefs[other][item]*sim
                # Sum of similarities
                simSums.setdefault(item,0)
                simSums[item]+=sim                
    # Create a normalized list
    rankings=[(total/simSums[item],item) for item,total in totals.items( )]    
    # Return a sorted list
    rankings.sort( )
    rankings.reverse( )
    return rankings

def transformPrefs(prefs):
    result={}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item,{})            
            # Flip item and person
            result[item][person]=prefs[person][item]
    return result

transformPrefs(userRatings)

def calculateSimilarItems(prefs,n=10):
    # Create a dictionary similar items
    result={}    
    # Invert the preference matrix to be item-centric
    itemPrefs=transformPrefs(prefs)
    for item in itemPrefs:
#        if c%100==0: print("%d / %d" % (c,len(itemPrefs)))
        scores=Matches(itemPrefs,item,n=n,similarity=distance)
        result[item]=scores 
    return result

def getRecommendedItems(prefs,itemMatch,user):
    userRatings=prefs[user]
    scores={}
    totalSim={}
    
    # Loop over items rated by this user
    for (item,rating) in userRatings.items( ):
        
        # Loop over items similar to this one
        for (similarity,item2) in itemMatch[item]:
            
            # Ignore if this user has already rated this item
            if item2 in userRatings: continue
                
            # Weighted sum of rating times similarity
            scores.setdefault(item2,0)
            scores[item2]+=similarity*rating
            
            # Sum of all the similarities
            totalSim.setdefault(item2,0)
            totalSim[item2]+=similarity
            
    # Divide each total score by total weighting to get an average
    rankings=[(score/totalSim[item],item) for item,score in scores.items( )]
    
    # Return the rankings from highest to lowest
    rankings.sort( )
    rankings.reverse( )
    return rankings


itemsim=calculateSimilarItems(userRatings)



def plotDistance(album1, album2):
    data=[]
    for i in userRatings.keys():
        try:
            data.append((i,userRatings[i][album1], userRatings[i][album2]))
        except:
            pass
    df=pd.DataFrame(data=data, columns = ['user', album1, album2])
    plt.scatter(df[album1],df[album2])
    plt.xlabel(album1)
    plt.ylabel(album2)
    for i,t in enumerate(df.user):
        plt.annotate(t,(df[album1][i], df[album2][i]))
    plt.show()
    print(df)
       
plotDistance('Abbey Road', 'Ziggy Stardust')
print("My recomendations:")
print( getRecommendedItems(userRatings, itemsim,'Dave'))








