import numpy as np
import matplotlib.pyplot as plt
import kernels
import random
import plotting

class Regression(object):

	def __init__(self, Xtest, Xtrain, Ytrain, add_noise=1, kernel=None, Ytest=None, normalize=True):
		self.Xtest = Xtest
		self.Xtrain = Xtrain
		self.Ytest = Ytest
		self.Ytrain = Ytrain
		self.add_noise = add_noise
		self.kernel = kernel		

		if normalize is True:
			# Normalize training inputs
			X_mu = np.vstack(np.mean(Xtrain, axis=0))
			X_s = np.vstack(Xtrain.std(axis=0))
			train_centred = Xtrain.T - X_mu
			train_div = train_centred/X_s                       
			self.Xtrain = train_div.T   
			# Normalize test inputs according to training inputs
			test_centred = self.Xtest.T - X_mu
			test_div = test_centred/X_s 
			self.Xtest = test_div.T
			# Centre training outputs 
			self.Y_mu = np.mean(Ytrain)
			self.Ytrain = Ytrain - self.Y_mu
		if Ytest is not None:
			self.Ytest = Ytest - self.Y_mu
		
		if kernel is None:
			import warnings
			warnings.warn("Kernel not specified, defaulting to RBF kernel...")
			self.kernel = kernels.RBF()
		
		# Compute posterior mean vector
		Xtrain_cov = self.kernel.compute_noisy(self.Xtrain, self.Xtrain)
 		cross_cov = self.kernel.compute(self.Xtest, self.Xtrain)
		tr_chol = np.linalg.cholesky(Xtrain_cov) 
		tr_chol_inv = np.linalg.inv(tr_chol)
		inv = np.dot(tr_chol_inv.T, tr_chol_inv) 
 		cross_x_inv = np.dot(cross_cov, inv)
 		post_mean = (np.dot(cross_x_inv, self.Ytrain)) 
		noise = add_noise*np.reshape([random.gauss(0, np.sqrt(self.kernel.noise_var)) for i in range(0,post_mean.shape[0])],(-1,1))
		self.post_mean = post_mean + noise

 		# Compute posterior standard deviation and uncertainty bounds
		test_cov = self.kernel.compute_noisy(self.Xtest, self.Xtest)
		self.test_cov = test_cov
 		cov_post = test_cov - np.dot(np.dot(cross_cov,inv),cross_cov.T)
		cov_post = cov_post + (self.kernel.noise_var*np.eye((cov_post.shape[0])))
		self.cov_post = cov_post
 		self.post_s = np.sqrt(np.diag(cov_post)).reshape(-1,1)
        
	def predict(self):   
		# Return the posterior mean and upper and lower 95% confidence bounds
		return self.post_mean, self.post_mean+(2*self.post_s), self.post_mean-(2*self.post_s)

	def plot_by_index(self):
		upper = (self.post_mean + (2*self.post_s)).flat
		lower = (self.post_mean - (2*self.post_s)).flat
		index = np.arange(1,(self.Xtest.shape[0]+1),1)

		Y = self.Ytest
		Y1 = self.Ytest
		Y2 = self.Ytest
		post_mean = self.post_mean
		Ytest = np.sort(self.Ytest, axis=0)
		upper = (np.array([upper for Y,upper in sorted(zip(Y,upper))])).flat
		lower = (np.array([lower for Y1,lower in sorted(zip(Y1,lower))])).flat
		post_mean = (np.array([post_mean for Y2,post_mean in sorted(zip(Y2,post_mean))]))
        
		# Plot index against posterior mean function, uncertainty and true test values
		fig = plt.figure()
		plt.xlim(0, max(index)+1) 
#		plt.ylim(-15,10)    
		plt.plot(index, Ytest, 'ro')
		plt.plot(index, post_mean, 'r--', lw=2)
		plt.fill_between(index, lower, upper, color='#87cefa')
		plt.show()		

	def r_squared(self):
		obs_mean = np.mean(self.Ytest)
		ss_tot = np.sum((self.Ytest-obs_mean)**2)
		ss_res = np.sum((self.Ytest-self.post_mean)**2)
		r_sq = 1 - (ss_res/ss_tot)
		return r_sq
    
	def plot_prior(self):
		if self.Xtrain.shape[1] == 1:
			plotting.plot_prior_1D(self.Xtest, self.test_cov, Ytest=self.Ytest)
		elif self.Xtrain.shape[1] == 2:
			plotting.plot_prior_2D(self.Xtest, self.test_cov, Ytest=self.Ytest)
		else:
			print "The dimensionality of the input space is too high to visualize."
        
	def plot_posterior(self):
		if self.Xtrain.shape[1] == 1:
			plotting.plot_posterior_1D(self.Xtest, self.Xtrain, self.Ytrain, self.post_mean, self.post_s, self.cov_post, Ytest=self.Ytest)
		elif self.Xtrain.shape[1] == 2:
			plotting.plot_posterior_2D(self.Xtest, self.Xtrain, self.Ytrain, self.post_mean, self.post_s, Ytest=self.Ytest)
		else:
			print "The dimensionality of the input space is too high to visualize. Use plot_by_index instead."

