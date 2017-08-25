import numpy as np
import pandas as pd

def sigmoid(Z):
	return 1/(1+np.exp(-Z))
	
"""sigmoid gradient function"""
def sigmoidGrad(Z):
	 return(sigmoid(Z)*(1-sigmoid(Z)))
	 
"""The neural network cost function is calculated in the following code"""
def nnComputeCost(params, ip_layr_size, hiddn_layr_size, num_labels, features, classes, reg):
	# unrolling theta to their respective dimensions
	theta1 = params[0:(hiddn_layr_size*(ip_layr_size+1))].reshape(hiddn_layr_size, ip_layr_size+1)	#25x401
	theta2 = params[hiddn_layr_size*(ip_layr_size+1):].reshape(num_labels, hiddn_layr_size+1)	#10:26
	
	#calculating the output of the neural network
	a2 = sigmoid(features.dot(theta1.T))
	a2 = np.c_[np.ones((a2.shape[0],1)), a2]
	a3 = sigmoid(a2.dot(theta2.T))		#5000x10
	
	#classes is a 5000x1 vector... make it a 5000x10 vector with 1s as the respective values of classes
	classes = pd.get_dummies(classes.ravel()).as_matrix()
	
	#calculating the cost of the neural network... Here y and log(hx) have simple multiplication not cross product... it also has a regularization term in it
	J = (-1/features.shape[0])* np.sum(np.log(a3)*(classes) + np.log(1-a3)*((1-classes))) + \
	reg/(2*features.shape[0])*(np.sum(np.square(theta1[:,1:])) + np.sum(np.square(theta2[1,1:])))
	
	
	return J
