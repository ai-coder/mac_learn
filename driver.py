import algorithms.linear_reg.linearreg as lr
import numpy as np
"""This is the main driver file to test the rest machine learning modules."""

#loads the data1_1.txt into data variable
data = np.genfromtxt("datas/data1_1.txt", delimiter=',')

#loading data to X and y - the first column corresponds to input and the second column corresponds to output
X=np.c_[np.ones(data.shape[0]),data[:,0]]
y=np.c_[data[:,1]]

theta =[[0],[0]]
J = lr.ComputeCost(X, y, theta)
print(J)
