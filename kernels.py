import numpy as np

class RBF(object):
    def compute(self, a, b):       	
	# sq_dist = square of sum of dist btwn two rows
	# returns exponential of squared distances matrix (with scaling factor)	
	sq_dist = np.sum(a**2, 1).reshape(-1, 1) + np.sum(b**2, 1) - 2*np.dot(a, b.T) 
        return np.exp(-.5 * (1/0.1) * sq_dist)





