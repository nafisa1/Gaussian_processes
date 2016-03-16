import numpy as np

#	Equivalent to:
#       a = np.linalg.norm(a, axis=1).reshape(-1,1)
#       b = np.linalg.norm(b, axis=1).reshape(-1,1)
#       cov = np.zeros((a.shape[0],b.shape[0]))
#       for i in range(0,a.shape[0]):
#           for j in range(0,b.shape[0]):
#               cov[i][j] = np.exp(-.5 * (1/0.1) * ((a[i]-b[j])**2))
#       return cov

class RBF(object):
    def __init__(self, lengthscale=1, sig_var=1, noise_var=1):
	self.lengthscale = lengthscale
	self.sig_var = sig_var
	self.noise_var = noise_var

    def compute(self, a, b):	       		
	sq_dist = np.sum(a**2, 1).reshape(-1, 1) + np.sum(b**2, 1) - 2*np.dot(a, b.T)
        cov = np.exp(-.5 * (1/(self.lengthscale**2)) * sq_dist)
	cov = (self.sig_var*cov)
	return cov

    def compute_noisy(self, a, b):	       		
	sq_dist = np.sum(a**2, 1).reshape(-1, 1) + np.sum(b**2, 1) - 2*np.dot(a, b.T)
        cov = np.exp(-.5 * (1/(self.lengthscale**2)) * sq_dist)
	noisy_cov = (self.sig_var*cov) + (self.noise_var*np.eye(cov.shape[1]))
	return noisy_cov
