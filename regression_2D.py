import numpy as np
import matplotlib.pyplot as plt
import kernels
from mpl_toolkits.mplot3d import Axes3D

class Regression(object):

	def __init__(self, X, Xn, Yn, noise_var, kernel=None, normalize=True):
		self.X = X
		self.Xn = Xn
		self.Yn = Yn
		self.kernel = kernel        
		self.noise_var = noise_var
		 
		if normalize is True: # works on 1D and 2D input
			mu = np.vstack(np.mean(Xn, axis=0))
			s = np.vstack(Xn.std(axis=0))
			centred = Xn.T - mu
			div = centred/s                       
			self.Xn = div.T   
			# Normalize test points according to training points
			test_centred = self.X.T - mu
			test_div = test_centred/s 
			self.X = test_div.T 
		
		if kernel is None:
			import warnings
			warnings.warn("Kernel not specified, defaulting to RBF kernel...")
			self.kernel = kernels.RBF()      

	def plot_prior(self):
  		# Calculate the standard deviation of the prior
		test_cov = self.kernel.compute(self.X, self.X)
 		noisy_cov = test_cov + (self.noise_var * np.eye(test_cov.shape[0]))
		s = np.sqrt(np.diag(noisy_cov))
  		
  		# Create mean vector and vectors bounding the 95% uncertainty region
 		n = self.X[:,0].shape[0]
 		mean = np.zeros(n)
  		un = mean + (2*s)        
  		unn = mean - (2*s) 
  		       
 		# Plot mean points and uncertainty
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1, projection = '3d')
 		ax.scatter(self.X[:,0], self.X[:,1], mean) 
		ax.scatter(self.X[:,0], self.X[:,1], un, c= 'r')
		ax.scatter(self.X[:,0], self.X[:,1], unn, c= 'r')
		plt.show() 

	def predict(self):   
		# Compute mean vector
		Xn_cov = self.kernel.compute(self.Xn, self.Xn) 
 		cross_cov = self.kernel.compute(self.X, self.Xn)
 		inv = np.linalg.inv(Xn_cov + (self.noise_var*np.eye(Xn_cov.shape[0]))) 
 		cross_x_inv = np.dot(cross_cov, inv)
 		mean = np.dot(cross_x_inv, self.Yn)
		
 		# Compute posterior standard deviation and uncertainty bounds
		test_cov = self.kernel.compute(self.X, self.X) 
 		cov_post = test_cov - np.dot(np.dot(cross_cov,inv),cross_cov.T)
 		s = np.sqrt(np.diag(cov_post)).reshape(-1,1)
 		un = mean + (2*s)        
 		unn = mean - (2*s)
		return mean, un, unn
        
	def plot_posterior(self):   
		# Compute mean vector
		Xn_cov = self.kernel.compute(self.Xn, self.Xn) 
 		cross_cov = self.kernel.compute(self.X, self.Xn)
 		inv = np.linalg.inv(Xn_cov + (self.noise_var*np.eye(Xn_cov.shape[0]))) 
 		cross_x_inv = np.dot(cross_cov, inv)
 		mean = np.dot(cross_x_inv, self.Yn)
		
 		# Compute posterior standard deviation and uncertainty bounds
		test_cov = self.kernel.compute(self.X, self.X) 
 		cov_post = test_cov - np.dot(np.dot(cross_cov,inv),cross_cov.T)
 		s = np.sqrt(np.diag(cov_post)).reshape(-1,1)
 		un = mean + (2*s)        
 		unn = mean - (2*s)
		
		# Plot mean points and uncertainty
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1, projection = '3d')
 		ax.scatter(self.X[:,0], self.X[:,1], mean) 
 		ax.scatter(self.X[:,0], self.X[:,1], un, c='r')
 		ax.scatter(self.X[:,0], self.X[:,1], unn, c='r')
 		ax.scatter(self.Xn[:,0], self.Xn[:,1], self.Yn, c='g',marker='^', s = 70)
		plt.show()
