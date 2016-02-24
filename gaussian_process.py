import numpy as np
import matplotlib.pyplot as plt
import kernels

#
# A simple Gaussian process
# Parameters:
#	X = original input range
#

class GP(object):
	def __init__(self, X, Y, Xn, Yn, kernel=None, normalize=True):
		self.X = X
		self.Y = Y
		self.Xn = Xn
		self.Yn = Yn
		self.kernel = kernel        
    
		if normalize is True: # works on 1D and 2D input
			mu = np.vstack(np.mean(X, axis=0))
			s = np.vstack(X.std(axis=0))
			centred = X.T - mu
			div = centred/s                       
			self.X = div.T    
		self.mean = mu
		self.sd = s
		if kernel is None: 
			import warnings
			warnings.warn("Kernel not specified, defaulting to RBF kernel...")
			kernel = kernels.RBF()

        	# Compute covariance matrix
		self.cov = kernel.compute(self.X, self.X)          
        
	def prior(self):   
		cov = self.cov  
		X = self.X # Resetting X to normalized inputs
		mean = np.zeros((X.shape[0],1))
		noisy_cov = self.cov + (0.00005 * np.eye(self.cov.shape[0]))
		print noisy_cov
		L = np.linalg.cholesky(noisy_cov)
		print L
		print np.dot(L, np.random.normal(size=(self.X.shape[0],10))) 
		print mean + np.dot(L, np.random.normal(size=(self.X.shape[0],10)))

	def plot_prior(self):          
		X = np.hstack(self.X) # for plotting
		mean = np.zeros(X.shape)
		noisy_cov = self.cov + (0.00005 * np.eye(self.cov.shape[0]))       
		s = np.sqrt(np.diag(noisy_cov))
		           
		plt.figure()
		plt.xlim(min(X), max(X))
		plt.ylim(min(mean-(2*s)-(s/2)), max(mean+(2*s)+(s/2)))       
		plt.plot(X, self.Y, 'b-', label='Y')
		plt.plot(X, mean, 'r--', lw=2, label='mean')
		plt.fill_between(X, mean-(2*s), mean+(2*s), color='#87cefa')
		plt.legend()
		plt.show() 

		L = np.linalg.cholesky(noisy_cov)
		mean = mean.reshape(X.shape[0],1)
		f = mean + np.dot(L, np.random.normal(size=(self.X.shape[0],10)))
		plt.figure()
		plt.xlim(min(X), max(X))
		plt.plot(X, f)
		plt.title('Ten samples')
		plt.show() 
        
	def plot_posterior(self, Y, Xn, Yn, normalize=True): 
		original_input = self.X
		if normalize is True: # works on 1D and 2D input
			centred = Xn.T - self.mean
			div = centred/self.sd                       
			Xn = div.T
		Xn_cov = self.kernel.compute(Xn, Xn)
		cross_cov = self.kernel.compute(Xn, original_input)
		Lx = np.linalg.cholesky(Xn_cov + (0.00005 * np.eye(Xn_cov.shape[0])))
		tr_oldcov_to_cross = np.linalg.solve(Lx, cross_cov)
		mu = np.dot(tr_oldcov_to_cross.T, np.linalg.solve(Lx, Yn))
		cov_post = self.cov + (0.00005*np.eye(self.cov.shape[0])) - np.dot(tr_oldcov_to_cross.T, tr_oldcov_to_cross)
		s = np.sqrt(np.diag(cov_post))
		mean = mu.flat
		X = np.hstack(original_input)

		plt.figure()
		plt.plot(Xn, Yn, 'r+', ms=20) # new points
		plt.xlim(min(X), max(X))
		plt.ylim(min(mean-(2*s)-(s/2)), max(mean+(2*s)+(s/2)))        
		plt.plot(X, Y, 'b-', label='Y') # true function
		plt.plot(X, mean, 'r--', lw=2, label='mean') # mean function
		plt.fill_between(X, mean-(2*s), mean+(2*s), color='#87cefa') # uncertainty
		plt.legend()
		plt.show()
		        
		L = np.linalg.cholesky(cov_post)
		mu = mu.reshape(X.shape[0],1)
		f = mu + np.dot(L, np.random.normal(size=(X.shape[0],10)))
		plt.figure()
		plt.xlim(min(X), max(X))
		plt.plot(X, f)
		plt.title('Ten samples')
		plt.show()
