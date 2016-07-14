import numpy as np
from utils import normalize_centre, centre
import GPy

class Model(object):

	def __init__(self, X, Y, latent_dim, have_Ytest=True, split_train=0.8, kernel=None):
		self.X = X
		self.Y = Y
		self.kernel = kernel
		self.have_Ytest = have_Ytest
		self.split_train = split_train

		# Sanity check
		if len(Y.shape) != 2:
			Y = Y.reshape(-1,1)
		print Y.shape
		# Normalise and centre X, perform PCA
		X = normalize_centre(X)
		X_pcs, W = GPy.util.linalg.pca(X, latent_dim)
	
		if have_Ytest == True:

			# Shuffle X and Y (still corresponding)
			p = np.random.permutation(X.shape[0])
			X_pcs = X_pcs[p]
			Y = Y[p]
					
			# Split
			self.Xtrain = X_pcs[:(split_train*X.shape[0]),:]
			Ytrain = Y[:(split_train*X.shape[0]),:]
			self.Xtest = X_pcs[(split_train*X.shape[0]):,:]
			Ytest = Y[(split_train*X.shape[0]):,:]

			# Centre Y
			self.Ytrain = centre(Ytrain)
			self.Ytest = centre(Ytrain, Ytest)

		else:

			# Split
			self.Xtrain = X_pcs[:Y.shape[0],:]
			Ytrain = Y
			self.Xtest = X_pcs[Y.shape[0]:,:]

			# Centre Y
			self.Ytrain = centre(Ytrain)

	def test(self):
		return self.Y, self.Ytrain, self.Ytest
