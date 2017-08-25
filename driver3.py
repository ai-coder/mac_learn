import numpy as np
from scipy.io import loadmat
import algorithms.neural_network.neuralnet as nn



"""The data stored here are in the dictionary format. weights.keys() gives the keys of the dictionary"""
weights = loadmat('datas/data3_w.mat')
data = loadmat('datas/data3_1.mat')

"""The shape of theta1 and theta2 will be 25x401 and 10x26 repectively. The Theta1 and Theta2 used here are keys of dictionary. The theta1 and theta2 are the optimized value of theta."""
y = data['y']
X = np.c_[np.ones((data['X'].shape[0],1)), data['X']]
theta1, theta2 = weights['Theta1'],weights['Theta2']

"""We follow the feed forward rule of the neural network to calculate the output of the network. Following formulas are the general formulas of the neural network. We add bias term to every layer except the output layer"""
a2 = nn.sigmoid(X.dot(theta1.T))
a2 = np.c_[np.ones((a2.shape[0], 1)), a2]
a3 = nn.sigmoid(a2.dot(theta2.T))

"""predicting the accuracy of the network"""
pred = np.argmax(a3, axis=1)+1
print('Training set acccuracy : {} %'.format(np.mean(pred== y.ravel())*100))

#===================================================================================================

"""rolling the theta1 and theta2 into a single dimension vector"""
params = np.r_[theta1.ravel(), theta2.ravel()]
J = nn.nnComputeCost(params, 400, 25, 10, X, y, 0)
print(J)
