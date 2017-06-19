import numpy as np

def distance(f1,f2):
    diff = f1 - f2
    return diff.T.dot(diff)

def getDistanceMatrix(X):
    mm = X.shape[0]
    nn = X.shape[1]
    distanceMatrix = np.nan * np.eye(mm)
    for i in range(0,mm):
        for j in range(0,mm):
            if (i==j):
                continue
            f1 = X[i,:]
            f2 = X[j,:]
            distanceMatrix[i,j] = distance(f1,f2)
    return distanceMatrix

def getDistanceMatrixTrainTest(train, test):
    mtrain = train.shape[0]
    mtest = test.shape[0]
    distanceMatrix = np.zeros((mtest,mtrain))
    for i in range(0,mtest):
        for j in range(0,mtrain):
            f1 = train[j,:]
            f2 = test[i,:]
            distanceMatrix[i,j] = distance(f1,f2)
    return distanceMatrix