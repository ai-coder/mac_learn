import numpy as np
"""This is the implementation of algorithms of linear regression"""

"""This function computes the cost of the data using the linear regression cost function formula. The .dot function is used for matrix multiplication"""

def ComputeCost(X, y, theta):
	J = 1/(2*y.size)*np.sum(np.square(X.dot(theta)-y))
	return J
