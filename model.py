import numpy as np
import utils
import GPy

class Model(object):

	def __init__(self, X, Y, latent_dim, have_Ytest=True, shuffle=True, split_train=0.8, kernel=None, prior=None, acq_func=None):
		self.X = X
		self.Y = Y
		self.kernel = kernel
		self.have_Ytest = have_Ytest
		self.split_train = split_train
		self.prior = prior
		self.acq_func = acq_func

		# Sanity check
		if len(Y.shape) != 2:
			Y = Y.reshape(-1,1)
	
		# Normalise and centre X, perform PCA
		X = utils.remove_zero_cols(X)
		X = utils.normalize_centre(X)
		X_pcs, W = GPy.util.linalg.pca(X, latent_dim)
		jitter = 0.05*np.random.rand((X_pcs.shape[0]), (X_pcs.shape[1]))
		jitter -= 0.025
		X_pcs -= jitter
	
		if have_Ytest == True:
			
			if shuffle == True:
				# Shuffle X and Y (still corresponding)
				p = np.random.permutation(X.shape[0])
				X_pcs = X_pcs[p]
				Y = Y[p]
				# Also need to shuffle and cull SMILES

				if prior is not None:
					prior = prior[p]
					
			# Split
			self.Xtrain = X_pcs[:(split_train*X.shape[0]),:]
			self.Ytrain = Y[:(split_train*X.shape[0]),:]
			self.Xtest = X_pcs[(split_train*X.shape[0]):,:]
			self.Ytest = Y[(split_train*X.shape[0]):,:]

			# Centre Y
#			self.Ytrain_mean = np.mean(Ytrain)
#			self.Ytrain = utils.centre(Ytrain)
#			self.Ytest = utils.centre(Ytrain, Ytest)

		else:

			# Split X
			self.Xtrain = X_pcs[:Y.shape[0],:]
			self.Xtest = X_pcs[Y.shape[0]:,:]

			# Centre Y
			self.Ytrain = utils.centre(Y)

		if prior is not None:
			prior = prior.reshape(-1,1)
			assert prior.shape[0] == X.shape[0], "There must be one prior mean value for each input value (regardless of whether it is for training or testing)."
			# have prior mean values for Xtrain and Xtest
			# subtract training prior mean from Ytrain values
			prior_train = prior[:self.Xtrain.shape[0],:]
			self.Ytrain = self.Ytrain - prior_train
			self.prior_test = prior[self.Xtrain.shape[0]:,:]

	def hyperparameters(self):
		lat_hyp = utils.LHS(self.kernel)

		Xtrain = self.Xtrain #[:((self.Xtrain.shape[0])*0.8),:]
		Xtest = self.Xtest #train[((self.Xtrain.shape[0])*0.8):,:]
		Ytrain = utils.centre(self.Ytrain)
		Ytest = utils.centre(self.Ytrain, self.Ytest)
		#Ytrain = self.Ytrain[:((self.Ytrain.shape[0])*0.8),:]
		#Ytest = self.Ytrain[((self.Ytrain.shape[0])*0.8):,:]

		self.kernel.lengthscale, self.kernel.sig_var, self.kernel.noise_var = lat_hyp.compute(Xtest, Xtrain, Ytrain, Ytest)

	def regression(self):
		import regression

		if self.have_Ytest == True:
			# Centre Y
			self.Ytrain_mean = np.mean(self.Ytrain)
			Ytrain = utils.centre(self.Ytrain)
			Ytest = utils.centre(self.Ytrain, self.Ytest)

			regress = regression.Regression(self.Xtest, self.Xtrain, Ytrain, kernel=self.kernel, Ytest=Ytest)

		else:
			regress = regression.Regression(self.Xtest, self.Xtrain, self.Ytrain, kernel=self.kernel, Ytest=None)

		return regress

	def optimization(self, plot=False):

		Ytrain = utils.centre(self.Ytrain)
		Ytest = utils.centre(self.Ytrain, self.Ytest)

		if plot==False:
			new_x, ind = self.acq_func.compute(self.Xtest, self.Xtrain, Ytrain, self.kernel, plot=False)
		else:
			new_x, ind = self.acq_func.compute(self.Xtest, self.Xtrain, Ytrain, self.kernel, plot=True)

		new_obs = self.Ytest[ind]		

		self.Xtrain = np.vstack((self.Xtrain, new_x))
		self.Ytrain = np.vstack((self.Ytrain, new_obs))

		self.Xtest = np.delete(self.Xtest, ind, axis=0)
		self.Ytest = np.delete(self.Ytest, ind, axis=0)

		return new_x

	# This is done after regression etc			
	def correction(self):
		assert prior is not None, "Posterior mean correction is not required for a zero mean prior."
		# add test prior mean to posterior mean
		self.Ytest = self.Ytest + self.prior_test
		# will also have to apply correction to random draws

	def classify(self, threshold):
		# separate into compounds below and above threshold
		assert have_Ytest == True, "Experimental output values for the test set are required to calculate ROC plot."
		# using real Ytest values, calculate true positives, false positives, true negatives, false negatives
		# ROC plot (add function to plotting module)
