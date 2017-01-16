import numpy as np
import utils
import GPy
import kernels

class Model(object):

	def __init__(self, Ytrain, Ytest, pca=False, latent_dim=None, Xtrain=None, Xtest=None, smiles_train=None, smiles_test=None, kernel=None, prior_train=None, prior_test=None, acq_func=None):

		self.latent_dim = latent_dim
		self.Xtrain = Xtrain
		self.Xtest = Xtest
		self.Ytrain = Ytrain
		self.Ytest = Ytest
		self.smiles_train = smiles_train
		self.smiles_test = smiles_test
		self.kernel = kernel
		self.prior_train = prior_train
		self.prior_test = prior_test
		self.acq_func = acq_func

		# Sanity check	
		if len(Ytrain.shape) != 2:
			self.Ytrain = Ytrain.reshape(-1,1)	
		if len(Ytest.shape) != 2:
			self.Ytest = Ytest.reshape(-1,1)

		if Xtrain is not None: #self.kernel.datatype == 'numerical':
			print Xtrain.shape, Xtest.shape
			Xtrain, Xtest = utils.remove_identical(Xtrain, Xtest)
			print Xtrain.shape, Xtest.shape

			# Normalise and centre X, perform PCA
			Xtrain_nc = utils.normalize_centre(Xtrain)
			print np.vstack(Xtrain_nc.std(axis=0))
			Xtest_nc = utils.normalize_centre(Xtrain, Xtest)
			print np.vstack(Xtest_nc.std(axis=0))

			if pca == True:
				Xtrain, W = GPy.util.linalg.pca(Xtrain_nc, self.latent_dim)
				jitter = 0.05*np.random.rand((Xtrain.shape[0]), (Xtrain.shape[1]))
				jitter -= 0.025
				self.Xtrain = Xtrain - jitter
	
				Xtest = np.dot(W,Xtest_nc.T).T
				jitter = 0.05*np.random.rand((Xtest.shape[0]), (Xtest.shape[1]))
				jitter -= 0.025
				self.Xtest = Xtest - jitter

			else:
				self.Xtrain = Xtrain_nc
				self.Xtest = Xtest_nc

		if prior_train is not None:
			prior_train = prior_train.reshape(-1,1)
			# subtract training prior mean from Ytrain values
			self.Ytrain = self.Ytrain - prior_train

	def hyperparameters(self):
		lat_hyp = utils.LHS(self.kernel)

		Xtrain = self.Xtrain
		Xtest = self.Xtest 
		Ytrain = utils.centre(self.Ytrain)
		Ytest = utils.centre(self.Ytrain, self.Ytest)

		self.kernel.lengthscale, self.kernel.sig_var, self.kernel.noise_var = lat_hyp.compute(Xtest, Xtrain, Ytrain, Ytest)

	def regression(self):
		import regression

#		log_Ytrain = np.log(self.Ytrain)
#		Ytrain_mean = np.mean(self.Ytrain)
		Ytrain = utils.centre(self.Ytrain)

#		log_Ytest = np.log(self.Ytest)
		Ytest = utils.centre(self.Ytrain, self.Ytest)

		if self.Xtrain is None:
			regress = regression.Regression(Ytrain, Ytest, smiles_train=self.smiles_train, smiles_test=self.smiles_test, kernel=self.kernel)

		elif self.smiles_train is None:
			regress = regression.Regression(Ytrain, Ytest, Xtrain=self.Xtrain, Xtest=self.Xtest, kernel=self.kernel)

		else:
			regress = regression.Regression(Ytrain, Ytest, Xtrain=self.Xtrain, Xtest=self.Xtest, smiles_train=self.smiles_train, smiles_test=self.smiles_test, kernel=self.kernel)

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
		# using real Ytest values, calculate true positives, false positives, true negatives, false negatives
		# ROC plot (add function to plotting module)
		pass
