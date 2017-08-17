import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.optimize import minimize

from sklearn.preprocessing import PolynomialFeatures

"""The following function is written to plot a two features x and a binary classification"""
def plotData(data,X, y, x_label, y_label, pos_label, neg_label, axes=None):
	#getting indexes for y=0 and y=1
	neg = y[:,0] == 0 
	pos = y[:,0] == 1
	if axes==None:
		axes = plt.gca()
	axes.scatter(X[pos][:,1], X[pos][:,2], marker='+', c='k', s=60, linewidth=2, label=pos_label)
	axes.scatter(X[neg][:,1], X[neg][:,2], c='y', s=60, label=neg_label)
	axes.set_xlabel(x_label)
	axes.set_ylabel(y_label)
	axes.legend(frameon = True, fancybox = True)
	plt.show()
