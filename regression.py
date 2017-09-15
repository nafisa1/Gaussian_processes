import numpy as np
import matplotlib.pyplot as plt
import kernels
import random
import plotting
import utils

class Regression(object):

	def __init__(self, Ytrain, kernel=kernels.RBF(), add_noise=0.001, print_jit=False, Ytest=None, Xtest=None, Xtrain=None, cent_threshold=None):
		self.Xtest = Xtest
		self.Xtrain = Xtrain
		self.Ytest = Ytest
		self.Ytrain = Ytrain
		self.add_noise = add_noise
		self.kernel = kernel		
		self.cent_threshold = cent_threshold
		self.print_jit = print_jit


		if len(self.Xtrain) == 2:
			for n,item in enumerate(self.Xtrain):
				if isinstance(item[0], str) == False:
					Xtrain_num = np.asarray(item).reshape(-1,len(item[0]))
					Xtest_num = np.asarray(self.Xtest[n]).reshape(-1,len(item[0]))

					Xtest_num = utils.normalize_centre(Xtrain_num, Xtest_num)
					Xtrain_num = utils.normalize_centre(Xtrain_num)

				else:
					Xtrain_smiles = item
					Xtest_smiles = self.Xtest[n]

		elif isinstance(self.Xtrain[0], str) == False:
			Xtrain = np.asarray(self.Xtrain).reshape(-1,1)
			Xtest = np.asarray(self.Xtest).reshape(-1,1)
			
			Xtest_num = utils.normalize_centre(Xtrain, Xtest)
			Xtrain_num = utils.normalize_centre(Xtrain)

#		if self.pca == True:
#			Xtrain_num, W = GPy.util.linalg.pca(Xtrain_num, self.latent_dim)
#			jitter = 0.05*np.random.rand((Xtrain_num.shape[0]), (Xtrain_num.shape[1]))
#			jitter -= 0.025
#			Xtrain_num = Xtrain_num - jitter
#	
#			Xtest_num = np.dot(W,Xtest_num.T).T
#			jitter = 0.05*np.random.rand((Xtest_num.shape[0]), (Xtest_num.shape[1]))
#			jitter -= 0.025
#			Xtest_num = Xtest_num - jitter


		if len(self.Xtrain) == 2:
			self.Xtrain = []
			self.Xtrain.append(Xtrain_num)
			self.Xtrain.append(Xtrain_smiles)
			self.Xtest = []
			self.Xtest.append(Xtest_num)
			self.Xtest.append(Xtest_smiles)
		elif isinstance(self.Xtrain[0], str) == False:
			self.Xtrain = Xtrain_num
			self.Xtest = Xtest_num
		elif isinstance(self.Xtrain[0], str) == True:
			self.Xtrain = self.Xtrain
			self.Xtest = self.Xtest




		# Compute posterior mean vector
		Xtrain_cov = self.kernel.compute(self.Xtrain, self.Xtrain, noise=True)
		train_test_cov = self.kernel.compute(self.Xtrain, self.Xtest)
#		print train_test_cov		
		tr_chol, jitter = kernels.jit_chol(Xtrain_cov, print_jit=self.print_jit) 
		trtecov_div_trchol = np.linalg.solve(tr_chol,train_test_cov)
 		ytr_div_trchol = np.linalg.solve(tr_chol,self.Ytrain)
 		post_mean = (np.dot(trtecov_div_trchol.T, ytr_div_trchol)).reshape(-1,1)
		noise = add_noise*np.reshape([random.gauss(0, np.sqrt(self.kernel.noise_var)) for i in range(0,post_mean.shape[0])],(-1,1))
		self.post_mean = post_mean + noise

 		# Compute posterior standard deviation and uncertainty bounds
		test_cov = self.kernel.compute(self.Xtest,self.Xtest)

 		self.cov_post = (test_cov) - np.dot(trtecov_div_trchol.T,trtecov_div_trchol)
 		self.post_s = np.sqrt(np.absolute(np.diag(self.cov_post))).reshape(-1,1)

	def predict(self):   
		# Return the posterior mean and upper and lower 95% confidence bounds
		return self.post_mean, self.post_mean+(2*self.post_s), self.post_mean-(2*self.post_s)

	def plot_by_index(self):
		upper = (self.post_mean + (2*self.post_s)).flat
		lower = (self.post_mean - (2*self.post_s)).flat
		index = np.arange(1,(self.Ytest.shape[0]+1),1)

		colours = []
		for i in xrange(len(self.Ytest)):
			if len(self.Xtrain) == 2:
				if self.Xtest[1][i] in self.Xtrain[1]:
					colours.append('y')
				else:    
					colours.append('r')
			else:
				if self.Xtest[i] in self.Xtrain:
					print self.Xtest[i]
					colours.append('y')
				else:    
					colours.append('r')
		Y = self.Ytest
		Y1 = self.Ytest
		Y2 = self.Ytest
		Y3 = self.Ytest
		post_mean = self.post_mean
		Ytest = np.sort(self.Ytest, axis=0)
		upper = (np.array([upper for Y,upper in sorted(zip(Y,upper))])).flat
		lower = (np.array([lower for Y1,lower in sorted(zip(Y1,lower))])).flat
		post_mean = (np.array([post_mean for Y2,post_mean in sorted(zip(Y2,post_mean))]))
		cmap = [colours for Y3,colours in sorted(zip(Y3,colours))]

		# Plot index against posterior mean function, uncertainty and true test values
		fig = plt.figure()
		plt.xlim(0, max(index)+1) 
		plt.xlabel('Compound')
		plt.ylabel('Centred output')
		
	
		plt.plot(index, post_mean, 'r--', lw=2)
		plt.fill_between(index, lower, upper, color='#87cefa')
		plt.scatter(index, Ytest, c=cmap, s=40)	
		if self.cent_threshold is not None:
			plt.plot([0, max(index)+1],[self.cent_threshold, self.cent_threshold])
		plt.show()		

	def r_squared(self):
		obs_mean = np.mean(self.Ytest)
		ss_tot = np.sum((self.Ytest-obs_mean)**2)
		ss_res = np.sum((self.Ytest-self.post_mean)**2)
		r_sq = 1 - (ss_res/ss_tot)
		return r_sq

	def classify(self): # ADD ROC PLOT, ENRICHMENT FACTORS
		assert self.cent_threshold is not None, "An active/inactive threshold is required for classification."
		true_positives, true_negatives = utils.classif(self.post_mean, self.Ytest, self.cent_threshold, roc=True)
		enrichment_factors = []
		Y2 = self.Ytest
		post_mean = (np.sort(self.post_mean, axis=0))[::-1]
		Ytest = (np.array([Y2 for self.post_mean,Y2 in sorted(zip(self.post_mean,Y2))]))[::-1]

		tpr = [0.0]
		fpr = [0.0]

		actives = 0.0
		inactives = 0.0
		for index,value in enumerate(post_mean):
			if Ytest[index] >= self.cent_threshold:
				actives += 1.0       
			else:
				inactives += 1.0
			tpr.append(actives/float(true_positives))
			fpr.append(inactives/float(true_negatives))
		print true_positives
		print actives
		print true_negatives
		print inactives
		fig = plt.figure()
		x= [0.0,1,0]
		plt.plot(x,x, linestyle='dashed', color='red', linewidth=2)
		plt.plot(fpr,tpr, 'g', linewidth=5)
		plt.show()
				
    
	def plot_prior(self): # UPDATE FOR SMILES
		if self.Xtrain.shape[1] == 1:
			plotting.plot_prior_1D(self.Xtest, self.test_cov, Ytest=self.Ytest)
		elif self.Xtrain.shape[1] == 2:
			plotting.plot_prior_2D(self.Xtest, self.test_cov, Ytest=self.Ytest)
		else:
			print "The dimensionality of the input space is too high to visualize."
        
	def plot_posterior(self): # UPDATE FOR SMILES
		if self.Xtrain.shape[1] == 1:
			plotting.plot_posterior_1D(self.Xtest, self.Xtrain, self.Ytrain, self.post_mean, self.post_s, self.cov_post, Ytest=self.Ytest)
		elif self.Xtrain.shape[1] == 2:
			plotting.plot_posterior_2D(self.Xtest, self.Xtrain, self.Ytrain, self.post_mean, self.post_s, Ytest=self.Ytest)
		else:
			print "The dimensionality of the input space is too high to visualize. Use plot_by_index instead."

