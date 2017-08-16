import algorithms.linear_reg.linearreg as lr
import algorithms.linear_reg.plotdata as pd
import numpy as np
import matplotlib.pyplot as plt

"""This is the main driver file to test the rest machine learning modules."""

#loads the data1_1.txt into data variable
#The data is a 97 x 2 matrix
data = np.genfromtxt("datas/data1_1.txt", delimiter=',')

#loading data to X and y - the first column corresponds to input and the second column corresponds to output
#X becomes a 97x2 matrix with first column as 1s and y becomes a 97x1 matrix
X=np.c_[np.ones(data.shape[0]),data[:,0]]
y=np.c_[data[:,1]]


#plotting X and y
pd.plotData(X, y, 'x-label', 'y-label')
plt.close()

#initialise theta to zeros - theta is a 2 x 1 matrix
#ComputeCost computes the cost of the linear regression
theta =[[0],[0]]
J = lr.ComputeCost(X, y, theta)
print(J)

iteration = 1500
alpha = 0.01

#theta is the optimized theta after 1500 iterations and J_plot is the values of 1500 cost with theta after every iteration
#theta is the same 2 x 1 matrix and J_plot has 1500 values in a array
theta, J_plot= lr.gradientDescent(X, y, theta, iteration, alpha)

#plotting the cost of hypothesis in 1500 iterations
pd.plotDataJ('x-label', 'y-label', J_plot)
#the optimised theta is
print(theta)

#new cost must decrease
J = lr.ComputeCost(X,y,theta)
print(J)

#plotting the predicted function i.e. theta0*x0 + theta1*x1
#set the range for which hypothesis is to be plotted
range_x = np.arange(30)
#calculating the formula of hypothesis with the new theta
hx = theta[0]+theta[1]*range_x
pd.plotDataH(X, y, 'x-label', 'y-label', hx, range_x)

#using scikit-learn to plot the same and comapring both the values
pd.plotDataSK(X, y, range_x, hx)













