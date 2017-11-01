import numpy as np


"""Defining sigmoid function"""
def sigmoid(Z):
	return 1/(1+np.exp(-Z))
	
"""computing cost for a given X and y in logistic regression i.e. binary classification"""
def computeCost(theta, X, y):
	J = (-1/y.size) * (np.log(sigmoid(X.dot(theta))).T.dot(y) + (np.log(1-sigmoid(X.dot(theta))).T.dot(1-y)))
	if np.isnan(J[0]):
		return(np.inf)
	grad = (1/y.size)*X.T.dot(sigmoid(X.dot(theta.reshape(-1,1))) - y)
	return(J, grad.flatten())
	#flatten converts the two dimentional array i.e. [[x],[y],[z]] into one dimensional i.e. [x, y, z]
	
"""To use the minimize function we need to code the function returning grad and theta separately. The following function is trying to implement cost function and grad descent separately so that the it is compatible with the minimize function in the driver1.py file"""

def cGrad(theta, X, y):
	m = y.size
	h = sigmoid(X.dot(theta.reshape(-1,1)))
	grad = (1/m)*X.T.dot(h-y)
	return(grad.flatten())
	
def cCost(theta, X,y):
	m = y.size
	h = sigmoid(X.dot(theta))
	J=-1*(1/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y))
	if np.isnan(J[0]):
		return(np.inf)
	return(J[0])
