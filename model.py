import numpy as np
from utils import normalize_centre, centre
import GPy
import matplotlib.pyplot as plt

class Model(object):

	def __init__(self, X, Y, latent_dim, have_Ytest=True, shuffle=True, split_train=0.8, kernel=None, prior=None):
		self.X = X
		self.Y = Y
		self.kernel = kernel
		self.have_Ytest = have_Ytest
		self.split_train = split_train
		self.prior = prior

		# Sanity check
		if len(Y.shape) != 2:
			Y = Y.reshape(-1,1)
	
		# Normalise and centre X, perform PCA
		X = normalize_centre(X)
		X_pcs, W = GPy.util.linalg.pca(X, latent_dim)
	
		if have_Ytest == True:
			
			if shuffle == True:
				# Shuffle X and Y (still corresponding)
				p = np.random.permutation(X.shape[0])
				X_pcs = X_pcs[p]
				Y = Y[p]

				if prior is not None:
					prior = prior[p]
					
			# Split
			self.Xtrain = X_pcs[:(split_train*X.shape[0]),:]
			Ytrain = Y[:(split_train*X.shape[0]),:]
			self.Xtest = X_pcs[(split_train*X.shape[0]):,:]
			Ytest = Y[(split_train*X.shape[0]):,:]

			# Centre Y
			self.Ytrain = centre(Ytrain)
			self.Ytest = centre(Ytrain, Ytest)

		else:

			# Split X
			self.Xtrain = X_pcs[:Y.shape[0],:]
			self.Xtest = X_pcs[Y.shape[0]:,:]

			# Centre Y
			self.Ytrain = centre(Y)

		if prior is not None:
			prior = prior.reshape(-1,1)
			assert prior.shape[0] == X.shape[0], "There must be one prior mean value for each input value (regardless of whether it is for training or testing)."
			# have prior mean values for Xtrain and Xtest
			# subtract training prior mean from Ytrain values
			prior_train = prior[:self.Xtrain.shape[0],:]
			self.Ytrain = self.Ytrain - prior_train
			self.prior_test = prior[self.Xtrain.shape[0]:,:]

	# This is done after regression etc			
	def correction(self):
		# add test prior mean to posterior mean
		self.Ytest = self.Ytest + self.prior_test
		# will also have to apply correction to random draws
