import numpy as np
import algorithms.logistic_reg.logisticreg as lr
import algorithms.logistic_reg.plotdata as pd
from scipy.optimize import minimize

#importing data which is 100x3  || to get the dimensions of a matrix we use data.shape
data=np.genfromtxt('datas/data2_1.txt', delimiter=',')

#dividing the data into input and output
# X is the input with 100x3 dimension and y is 100x1
X = np.c_[np.ones(data.shape[0]), data[:,0:2]]
y = np.c_[data[:,2]]

#plotting the data
pd.plotData(data ,X, y, 'x-label', 'y-label', 'positive', 'negative')

#calculating cost at theta as 0s and gradient
#theta = [[0],[0],[0]]
theta = np.zeros(X.shape[1])
J, grad = lr.computeCost(theta, X, y)
print('The cost of the given data with theta as 0s is : ')
print(J)
print('The gradient is : ')
print(grad)

"""Improve the fucntions in the algorithm folder in order to make this function comaptible with them"""
#res = minimize(lr.cCost, theta, args=(X,y), method=None, jac=lr.cGrad, options={'maxiter':400})
