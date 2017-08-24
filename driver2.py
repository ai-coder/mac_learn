import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import algorithms.multiclass_classif.plotdata as pd
import algorithms.multiclass_classif.multiclassif as mc
from scipy.io import loadmat
from scipy.optimize import minimize



""" loading data from matlab file where datas are stored in form of dictionary. You can use data.keys() or weights.keys() to get the list of the keys in the dictionary."""

data = loadmat('datas/data3_1.mat')
weights = loadmat('datas/data3_w.mat')

"""The shape of X wnd y will be 5000 x 401 and 5000 x 1 repectively after the following line of assignment. The shape of theta1 and theta2 will be 25x401 and 10x26 repectively. The X y Theta1 and Theta2 used here are keys of the dictionary format data stored in the matlab file"""

y = data['y']
X = np.c_[np.ones((data['X'].shape[0],1)), data['X']]
theta1, theta2 = weights['Theta1'],weights['Theta2']

"""The following function is used to show the data present in X. 30 is used to represent the number of data you need to see. Replace it with any number"""
#pd.showData(X, 30)

"""The following functions is used to find the optimal value of theta and stone in the theta matrix"""
theta = mc.getTheta(X,y,10,0.1)

"""Calculating accuracy in predictions on the training set itself"""
probs = mc.sigmoid(X.dot(theta.T))
pred = np.argmax(probs, axis=1)+1
print('Training set accuracy : {} %'.format(np.mean(pred==y.ravel())*100))


"""Using scikit-learn to predict multiple logistic regression"""
#contributers are welcomed to write the details of the method scikit-learn uses to optimize data
clf = mc.sklTheta(X,y)
pred2 = clf.predict(X[:,1:])
print('Training set accuracy : {} %'.format(np.mean(pred2==y.ravel())*100))
