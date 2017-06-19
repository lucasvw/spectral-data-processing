import numpy as np
import matplotlib.pyplot as plt
from weight import weight

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

tau = 5

y_hat = np.zeros(mm)
for i in range(0, mm):
    W = weight(x2, x2[i], tau)
    XtWX = (X.T).dot(W).dot(X)
    XtWX_inv = np.linalg.inv(XtWX)
    theta_w = (XtWX_inv).dot(X.T).dot(W).dot(y)
    y_hat[i] = X.dot(theta_w)[i]

plt.figure(1, figsize=(8, 4))
plt.ylabel('Intensity')
plt.xlabel("Wavelength")
plt.title("Smoothing some spectral data with WLS regression")
plt.scatter(x2, y, marker="x", c="red", s=2 )
plt.plot(x2, y_hat, "b-")
plt.legend()
plt.show()
