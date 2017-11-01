import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from sklearn.linear_model import LinearRegression
from . import linearreg as lr

"""The following function plots X and y on both the axis. plt.scatter plots the data, s gives the size of the marker, c gives the color of the marker and marker='x' gives the pointer style. We have passed strings as x_label and y_label"""

def plotData(X, y, x_label, y_label):
	plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1)
	plt.xlim(4,24);
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.show()
	
"""plotting values of J in every iteration - total number of iteration is 1500"""
def plotDataJ(x_label, y_label, J):
	plt.plot(J)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.show()

"""plotting the hypothesis with calulated theta(optimised)"""	
def plotDataH(X,y,x_label, y_label, hx, range_x):
	plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1)
	plt.plot(range_x, hx)
	plt.show()
	
"""plotting the hypothesis calculated using Scikit-Learn"""
def plotDataSK(X, y, range_x, hx):
	plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1)
	plt.plot(range_x, hx, label='Linear Regression with simple gradient descent')
	regr=LinearRegression()
	regr.fit(X[:,1].reshape(-1,1), y.ravel())
	plt.plot(range_x, regr.intercept_+regr.coef_*range_x, label='Linear Regression with Scikit Learn')
	plt.legend(loc=4)
	plt.show()

"""plotting cost for some 2 dimensional domain. linspace defines the domain for which we have to plot i.e. x-axis and y-axis of the graph."""	
def plotDataCOST(X, y):
	theta_0 = np.linspace(-10,10,50)
	theta_1 = np.linspace(-1,4,50)
	J_temp = np.zeros((theta_0.size,theta_1.size))
	for i in range(50):
		for j in range(50):
			theta=[[theta_0[i]],[theta_1[j]]]
			J_temp[i,j]=lr.ComputeCost(X, y, theta)
	#This is the new way of plotting 3d graphs in matplotlib
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(theta_0, theta_1, J_temp, rstride=1, cstride=1, alpha=1, cmap=plt.cm.jet)
	plt.show()
