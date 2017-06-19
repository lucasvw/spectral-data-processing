import numpy as np
import matplotlib.pyplot as plt
from weight import weight
from distance import getDistanceMatrix
from distance import getDistanceMatrixTrainTest
from distance import distance

def ker(t):
    return np.max([1-t, 0])

trainCSV = np.genfromtxt('quasar_train.csv',delimiter=",")
lambdas = trainCSV[0,:]
lambdas = lambdas.reshape((lambdas.shape[0],1))

train = np.genfromtxt('quasar_train_smooth.csv',delimiter=",")
test = np.genfromtxt('quasar_test_smooth.csv',delimiter=",")

right_index = np.where(lambdas == 1300)[0][0]
left_index = np.where(lambdas == 1200)[0][0]

lambdas_right = lambdas[right_index:,:]
lambdas_left = lambdas[:left_index,:]

train_right = train[:,right_index:]
test_right = test[:,right_index:]
train_left = train[:,:left_index]
test_left = test[:,:left_index]

mm = train.shape[0]

neighborhood_size = 3

mm_test = test.shape[0]
distanceM = getDistanceMatrixTrainTest(train_right, test_right)
test_left_hat = np.zeros(test_left.shape)
error = np.zeros((mm_test,1))
for i in range(0,mm_test):
    indices_of_neighbors = distanceM[i].argsort()[:neighborhood_size]
    distance_with_neighbors = distanceM[i][indices_of_neighbors]
    h = np.nanmax(distanceM[i])
    upper = 0
    lower = 0
    for j in range(0, neighborhood_size):
        kernel = ker(distance_with_neighbors[j] / h)
        upper = upper + kernel * train_left[indices_of_neighbors[j],:]
        lower = lower + kernel
        test_left_hat[i] = upper / lower
    error[i] = distance(test_left[i], test_left_hat[i])

print(np.average(error))


plt.figure(1, figsize=(4, 4))
plt.plot(lambdas, test[0], ".", label="true" )
plt.plot(lambdas_left, test_left_hat[0], ".", label="true" )
plt.legend()
plt.show()


