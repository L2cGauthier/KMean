# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import KMeans as km

def GenerateNRandTestData(shape=(100, 5)):
    """
    Generate a random matrix (with normally distributed values) of the passed shape.
    """
    testData = pd.DataFrame(np.random.randn(shape[0], shape[1]))
    return testData

def Generate3ClustersDataSet(shape=(400, 5)):
    """
    Generate a random matrix with 3 clusters more or less distinguishable
    """
    testData = np.random.randn(shape[0], shape[1])
    testData[100:199,:] = 2* np.random.randn(99, shape[1]) +8
    testData[200:299,:] = -4 * np.random.randn(99, shape[1]) -16
    testData[300:399, :] = -1 * np.random.randn(99, shape[1]) -2
    
    return pd.DataFrame(testData)

def SaveTestData(testData, path='Example/testSet.csv'):
    """
    Returns True if the data passed were successfully saved in the Example folder. Else, it returns False.
    """
    try:
        testData.to_csv(path, mode='w+', sep=',', index = False, header=False)
    except Exception as e:
        print("Could not save test data.")
        print(e)
        return False
    return True

def ReadTestData(path='Example/testSet.csv'):
    try:
        testData = np.genfromtxt(path, delimiter=',')
    except Exception as e:
        print(e)
        return 0
    return testData
    
if __name__ == "__main__":
    
    testData = np.genfromtxt("Example/projectedTestSet.csv", delimiter=',')
    numberOfIteration ,clusteredList = km.KMeans(testData, 3)
    
    

