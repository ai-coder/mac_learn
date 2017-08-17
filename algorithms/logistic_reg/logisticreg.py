import numpy as np


"""Defining sigmoid function"""
def sigmoid(Z):
	return 1/(1+np.exp(-Z))
	
"""computing cost for a given X and y in logistic regression i.e. binary classification"""
def computeCost(X, y, theta):
	J = (-1/y.size) * (np.log(sigmoid(X.dot(theta))).T.dot(y) + (np.log(1-sigmoid(X.dot(theta))).T.dot(1-y)))
	return(J, 0)
