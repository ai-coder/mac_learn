import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from sklearn.linear_model import LinearRegression

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
	
def plotDataH(X,y,x_label, y_label, hx, range_x):
	plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1)
	plt.plot(range_x, hx)
	plt.show()
	
def plotDataSK(X, y, range_x, hx):
	plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1)
	plt.plot(range_x, hx, label='Linear Regression with simple gradient descent')
	regr=LinearRegression()
	regr.fit(X[:,1].reshape(-1,1), y.ravel())
	plt.plot(range_x, regr.intercept_+regr.coef_*range_x, label='Linear Regression with Scikit Learn')
	plt.legend(loc=4)
	plt.show()
