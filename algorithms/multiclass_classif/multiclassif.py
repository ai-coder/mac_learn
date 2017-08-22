import numpy as np
from scipy.optimize import minimize

def sigmoid(Z):
	return 1/(1+np.exp(-Z))

"""The following function computes the cost of the given data with provided theta. The function also takes into account regularization. lam variable is used to set the value of lambda"""

def lrcomputeCostReg(theta, lam, X, y):
	h = sigmoid(X.dot(theta))
	m = y.size
	J = (-1/m)*(np.log(h).T.dot(y) + np.log(1-h).T.dot(1-y)) + (lam/(2*m))*np.sum(np.square(theta[1:]))
	if np.isnan(J[0]):
		return(np.inf)
	return(J[0])

"""The following function is intended to calculate gradient of the logistic regression with regularization"""
def lrGradReg(theta, lam, X, y):
	m = y.size
	h = sigmoid(X.dot(theta.reshape(-1,1)))
	grad = (1/m)*X.T.dot(h-y) + (lam/m)*np.r_[[[0]],theta[1:].reshape(-1,1)]
	return(grad.flatten())
	
"""The following fucntion uses advanced optimization to find out the value of theta for each and every labels(outputs). features is the input and classes is the given output. n_labels is the number of different classes and lam is lambda used for regularizaion"""
def getTheta(features, classes, n_labels, lam):
	theta = np.zeros((features.shape[1],1))	# 401x1
	all_theta = np.zeros((n_labels,features.shape[1]))	#10x401
	
	for c in np.arange(1,n_labels+1):
		res = minimize(lrcomputeCostReg, theta, args=(lam, features, (classes==c)*1), method = None, jac = lrGradReg, options={'maxiter':50})
		all_theta[c-1] = res.x
	return(all_theta)
	
