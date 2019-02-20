# -*- coding: utf-8 -*-
"""
K-means algorithm from scratch
"""
#_____________________________________________________
#IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from cycler import cycler 

#_____________________________________________________
#TOOLBOX FUNCTIONS
def EuclideanDistance(v1,v2):
    """
    Returns the distance between to Euclidean Vectors (v1 and v2). For this application,
    we assume that v1 and v2 are vectors of the same dimension.
    """
    if len(v1)==len(v2):
        sumSquaredBuffer = 0
        for i in range(len(v1)):
            sumSquaredBuffer += (v1[i]-v2[i])**2
        return sqrt(sumSquaredBuffer)       
    else:
        raise Exception("The 2 vectors passed don't have the same dimension ("+str(len(v1)) + " & "+str( len(v2))+")")
        
def ListAverage(l):
    """
    Returns the average of a list of numerical values.
    """
    buffer = 0
    for i in range(0,len(l)):
        buffer += float(l[i]) 
    return buffer/len(l)

def IndexOfSmallestListElt(l):
    """
    Returns the index of the smallest element of l.
    l has to have more than 1 element i.e. we have to look for at least 2 clusters which make sense.
    """
    minElt = l[0]
    minIndex = 0
    for i,elt in enumerate(l):
        if elt < minElt:
            minElt = elt
            minIndex = i
    return minIndex  

def RearrangeList(l):
    """
    Not very elegant thing. Takes a list of points (which are np.array with len==2)   
    and returns an np.array which first column is points x value, and second y value.
    """
    if len(l)!=0: #At some point, we might have an empty cluster
        rearrangedArray = np.array([l[0][0], l[0][1]],ndmin=2)
        skip = True
        
        for index, elt in enumerate(l):
            if skip:
                skip=False #Skip first point, we used it to initialise rearrangedArray
            else:
                rearrangedArray = np.vstack([rearrangedArray, [l[index][0], l[index][1]]])
        return rearrangedArray
    return []

def HaveSameInternalValues (l1, l2):
    """
    Work around the fact you can't test egality of 2 list of list of np.array.
    Only used for representation. Again, not very elegant, a better data structure would
    probably solve it, but as it is not used for classification but for representation
    it's not worth it.
    """
    if len(l1)==len(l2):
        for i, (elt1, elt2) in enumerate(zip(l1, l2)):
            if len(elt1)==len(elt2):
                for j, (elt3, elt4) in enumerate(zip(elt1, elt2)):
                    if len(elt3) == len(elt4):
                        for L, (elt5, elt6) in enumerate(zip(elt3, elt4)):
                            if elt5 == elt6:
                                return True
                            return False
                    return False
            return False
    return False

def ShowCurrentState(currentClusteredList, currentClusterCenters,k):
    """
    ShowCurrentState print the current state of the k-means algorithm as a plot where the current cluster center
    is represented in black, and each current cluster is represented with a different color.
    """
    #Cycle on 20 distinguishable colors for clusters
    plt.rc('axes', prop_cycle=(cycler('color', ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080'])))
            
    for p in range(k):
        plt.scatter(RearrangeList(currentClusteredList[p])[0:len(RearrangeList(currentClusteredList[p])),0],
                    RearrangeList(currentClusteredList[p])[0:len(RearrangeList(currentClusteredList[p])),1],
                    s=20)
        plt.scatter(currentClusterCenters[p][0],
                    currentClusterCenters[p][1],
                    s=100, c = '#000000')
    plt.show()
    
#_____________________________________________________
#KMEANS ALGORITHM
    
def KMeans (data, k):
    """
    Returns a tuple made of the number of iteration and a list of length k (each index associated with a cluster), 
    containing arrays of the coordinates of the data point associated with a cluster.
    """
    
    """Initialisation"""
    #Select k random points in the dataset   
    startingPoints = np.random.randint(low=0, high=len(data)-1, size=k)
    currentClusterCenters = []
    
    #currentClusteredList = np.empty(shape = (1,k), dtype=np.int8)
    oldClusteredList = []
    currentClusteredList = []
    for j in range(k):
        currentClusteredList.append([])
        oldClusteredList.append([])
        currentClusterCenters.append([data[startingPoints[j],0], data[startingPoints[j],1]])
    
    done=False
    iterCount = 1
    
    """Iterative sorting"""
    while done==False:
        currentClusteredList = []
        for j in range(k):
            currentClusteredList.append([])
        for i, elt in  enumerate(data):
            currentPoint = [data[i,0], data[i,1]]
            distanceList = []
            for j in range(k):
                currentClusterCenter = currentClusterCenters[j]
                distanceList.append(EuclideanDistance(currentPoint, currentClusterCenter ))
            currentClusteredList[IndexOfSmallestListElt(distanceList)].append(elt)
            
        if HaveSameInternalValues(currentClusteredList, oldClusteredList): 
            #ShowCurrentState(currentClusteredList, currentClusterCenters, k)
            print('\r(final iteration)')
            done = True
            
        else:
            ShowCurrentState(currentClusteredList, currentClusterCenters, k)
            oldClusteredList = currentClusteredList
            print('Iteration number',iterCount)
            iterCount+=1
                  
            for l, center in enumerate(currentClusterCenters):
                center[0] = ListAverage(RearrangeList(currentClusteredList[l])[0:len(RearrangeList(currentClusteredList[l])),0])
                center[1] = ListAverage(RearrangeList(currentClusteredList[l])[0:len(RearrangeList(currentClusteredList[l])),1])
    
    return (iterCount-1, currentClusteredList)
