import numpy as np
import matplotlib.pyplot as plt

trainCSV = np.genfromtxt('quasar_train.csv',delimiter=",")
lambdas = trainCSV[0,:]
train = trainCSV[1:,:]
test = np.genfromtxt('quasar_test.csv',delimiter=",",skip_header=1)

mm = lambdas.size

y = train[0,:]
y.reshape(y.size,1)

x1 = np.ones((mm,1))
x2 = lambdas.reshape((mm,1))

X = np.hstack((x1,x2))

XtX = (X.T).dot(X)
XtX_inv = np.linalg.inv( XtX )

theta = (XtX_inv).dot(X.T).dot(y)

plt.figure(1, figsize=(8, 4))
plt.scatter( x2, y, marker="x", c="red", s=2 )
plt.ylabel('Intensity')
plt.xlabel("Wavelength")
plt.title("Some noisy spectrum with an OLS regression")
plt.plot( x2, X.dot(theta), "b-", )
plt.show()
