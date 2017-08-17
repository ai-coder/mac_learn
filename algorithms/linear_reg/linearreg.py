import numpy as np
"""This is the implementation of algorithms of linear regression"""

"""This function computes the cost of the data using the linear regression cost function formula. The .dot function is used for matrix multiplication"""

def ComputeCost(X, y, theta):
	J = 1/(2*y.size)*np.sum(np.square(X.dot(theta)-y))
	return J
	
"""This function calculates the most suitable theta for the given dataset to have lowest cost"""
def gradientDescent(X, y, theta, iteration, alpha):
	J_temp = np.zeros(iteration)
	for i in range(iteration):
		theta = theta - alpha*(1/y.size)*(X.T.dot(X.dot(theta)-y))
		J_temp[i]=ComputeCost(X,y,theta)
	return(theta, J_temp)
	
	
"""References for the usage. Find the used functions or operator here"""

"""
1. .T operator - transpose operator
	[1, 2, 3, 4] return itself after transpose because it is considered as a single dimension in numpy array
	[[1, 2, 3, 4]] returns [[1],[2],[3],[4]] after its transpose because it is considered as a two dimensional array in numpy

2. .dot operator - matrix multiplication operator

"""
