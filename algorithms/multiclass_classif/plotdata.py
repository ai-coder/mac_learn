import matplotlib.pyplot as plt
import numpy as np

"""The following function is used to show the given data in the X. The data are in the structure of numeric photos which is plotted here and modified accordingly. Modify this function as per your data. The num is the nnumber of data to be shown"""

def showData(X, num):
	sample = np.random.choice(X.shape[0],num)
	plt.imshow(X[sample,1:].reshape(-1,20).T)
	plt.axis('off')
	plt.show()
