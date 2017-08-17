import numpy as np
import algorithms.logistic_reg.logisticreg as lr
import algorithms.logistic_reg.plotdata as pd

#importing data which is 100x3  || to get the dimensions of a matrix we use data.shape
data=np.genfromtxt('datas/data2_1.txt', delimiter=',')

#dividing the data into input and output
# X is the input with 100x3 dimension and y is 100x1
X = np.c_[np.ones(data.shape[0]), data[:,0:2]]
y = np.c_[data[:,2]]

#plotting the data
pd.plotData(data ,X, y, 'x-label', 'y-label', 'positive', 'negative')

#calculating cost at theta as 0s
theta = [[0],[0],[0]]
J, grad = lr.computeCost(X, y, theta)
print('The cost of the given data with theta as 0s is : ')
print(J)
