import numpy as np
import matplotlib.pyplot as plt
import kernels
import random
import plotting

class Regression(object):

	def __init__(self, Ytrain, Ytest, kernel=kernels.RBF(), add_noise=0.01, Xtest=None, Xtrain=None, smiles_train=None, smiles_test=None):
		self.Xtest = Xtest
		self.Xtrain = Xtrain
		self.smiles_train = smiles_train
		self.smiles_test = smiles_test
		self.Ytest = Ytest
		self.Ytrain = Ytrain
		self.add_noise = add_noise
		self.kernel = kernel		
		
		# Compute posterior mean vector
		if isinstance(self.kernel, kernels.Composite):
			if self.Xtrain is not None and self.smiles_train is not None:
				Xtrain_cov = self.kernel.compute_noisy(numA=self.Xtrain, numB=self.Xtrain, smilesA=self.smiles_train, smilesB=self.smiles_train)
 				cross_cov = self.kernel.compute(numA=self.Xtest, numB=self.Xtrain, smilesA=self.smiles_test, smilesB=self.smiles_train)
			
			elif self.Xtrain is None and self.smiles_train is not None:
				Xtrain_cov = self.kernel.compute_noisy(smilesA=self.smiles_train, smilesB=self.smiles_train)
 				cross_cov = self.kernel.compute(smilesA=self.smiles_test, smilesB=self.smiles_train)

			elif self.Xtrain is not None and self.smiles_train is None:
				Xtrain_cov = self.kernel.compute_noisy(numA=self.Xtrain, numB=self.Xtrain)
 				cross_cov = self.kernel.compute(numA=self.Xtest, numB=self.Xtrain)

		elif self.Xtrain is not None:
			Xtrain_cov = self.kernel.compute_noisy(self.Xtrain, self.Xtrain)
 			cross_cov = self.kernel.compute(self.Xtest, self.Xtrain)

		else:
			Xtrain_cov = self.kernel.compute_noisy(self.smiles_train, self.smiles_train)
 			cross_cov = self.kernel.compute(self.smiles_test, self.smiles_train)
		
		print np.linalg.det(Xtrain_cov)
		if np.linalg.det(Xtrain_cov) <= 0:
			Xtrain_cov = Xtrain_cov + (0.01*np.eye(Xtrain_cov.shape[1]))
		print np.linalg.det(Xtrain_cov)
		tr_chol = np.linalg.cholesky(Xtrain_cov) 
		tr_chol_inv = np.linalg.inv(tr_chol)
		inv = np.dot(tr_chol_inv.T, tr_chol_inv)
 		cross_x_inv = np.dot(cross_cov, inv)
 		post_mean = (np.dot(cross_x_inv, self.Ytrain))
		noise = add_noise*np.reshape([random.gauss(0, np.sqrt(self.kernel.noise_var)) for i in range(0,post_mean.shape[0])],(-1,1))
		self.post_mean = post_mean + noise

 		# Compute posterior standard deviation and uncertainty bounds
		if isinstance(self.kernel, kernels.Composite):
			if self.Xtrain is not None and self.smiles_train is not None:
				test_cov = self.kernel.compute(numA=self.Xtest, numB=self.Xtest, smilesA=self.smiles_test, smilesB=self.smiles_test)#compute_noisy(numA=self.Xtest, numB=self.Xtest, smilesA=self.smiles_test, smilesB=self.smiles_test)
			
			elif self.Xtrain is None and self.smiles_train is not None:
				test_cov = self.kernel.compute(smilesA=self.smiles_test, smilesB=self.smiles_test)#compute_noisy(smilesA=self.smiles_test, smilesB=self.smiles_test)

			elif self.Xtrain is not None and self.smiles_train is None:
				test_cov = self.kernel.compute(numA=self.Xtest, numB=self.Xtest)#compute_noisy(numA=self.Xtest, numB=self.Xtest)

		elif self.Xtrain is not None:
			test_cov = self.kernel.compute(self.Xtest,self.Xtest)#compute_noisy(self.Xtest, self.Xtest)

		else:
			test_cov = self.kernel.compute(self.smiles_test, self.smiles_test)#compute_noisy(self.smiles_test, self.smiles_test)
	
 		self.cov_post = test_cov - np.dot(np.dot(cross_cov,inv),cross_cov.T)
 		self.post_s = np.sqrt(np.diag(self.cov_post)).reshape(-1,1)

	def predict(self):   
		# Return the posterior mean and upper and lower 95% confidence bounds
		return self.post_mean, self.post_mean+(2*self.post_s), self.post_mean-(2*self.post_s)

	def plot_by_index(self):
		upper = (self.post_mean + (2*self.post_s)).flat
		lower = (self.post_mean - (2*self.post_s)).flat
		index = np.arange(1,(self.Ytest.shape[0]+1),1)

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
		plt.xlabel('Compound')
		plt.ylabel('Centred pIC50')    
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

