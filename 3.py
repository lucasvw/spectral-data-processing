import numpy as np
import matplotlib.pyplot as plt
from weight import weight

trainCSV = np.genfromtxt('quasar_train.csv',delimiter=",")
lambdas = trainCSV[0,:]
train = trainCSV[1:,:]
test = np.genfromtxt('quasar_test.csv',delimiter=",",skip_header=1)

mm = lambdas.size

x1 = np.ones((mm,1))
x2 = lambdas.reshape((mm,1))
X = np.hstack((x1,x2))

tau = 5

test_smooth = np.zeros(test.shape)
for j in range(0, test.shape[0]):
    y_hat = np.zeros(mm)
    y = test[j, :]
    y.reshape(y.size, 1)
    for i in range(0, mm):
        W = weight(x2, x2[i], tau)
        XtWX = (X.T).dot(W).dot(X)
        XtWX_inv = np.linalg.inv(XtWX)
        theta_w = (XtWX_inv).dot(X.T).dot(W).dot(y)
        y_hat[i] = X.dot(theta_w)[i]
    print("{0}% complete".format(j/test.shape[0]*100))
    test_smooth[j] = y_hat

np.savetxt("quasar_test_smooth.csv", test_smooth, delimiter=",")