import numpy as np
import utils
import GPy
import kernels

class Model(object):

	def __init__(self, pca=False, print_jit=False, latent_dim=None, X=None, Y=None, Ytrain=None, Ytest=None, Xtrain=None, Xtest=None, smiles_train=None, smiles_test=None, kernel=None, prior_train=None, prior_test=None, acq_func=None, threshold=None):

		self.pca = pca
		self.print_jit = print_jit
		self.latent_dim = latent_dim
		self.X = X
		self.Y = Y
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
		self.threshold = threshold

		# Sanity check
		if Y is not None:
			if len(Y.shape) != 2:
				self.Y = Y.reshape(-1,1)
		else:
			if len(Ytrain.shape) != 2:
				self.Ytrain = Ytrain.reshape(-1,1)	
			if len(Ytest.shape) != 2:
				self.Ytest = Ytest.reshape(-1,1)

		if Xtrain is not None: 
			Xtrain = np.asarray(Xtrain)
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

	def cross_validation(self):
		pass

	def max_log_likelihood(self, kernel):
		final_points = []
		log_likelihoods = []
		for choice in self.hparameter_choices:
			centred_cv_y = utils.centre(self.cv_y.reshape(-1,1))
			find_max_ll = max_likelihood.Max_LL(centred_cv_y, kernel)
			starting_point = []
			start.append(choice[0])
			start.append(choice[1])
			final_point, ll = find_max_ll.run_opt(starting_point)
			print final_point, ll
			final_points.append(final_point)
			log_likelihoods.append(ll)
		index = np.argmax(log_likelihoods)
		best_hparams = final_points[index]

	def hyperparameters(self):
		lat_hyp = utils.LHS(self.kernel)

		Ytrain = utils.centre(self.Ytrain)
		Ytest = utils.centre(self.Ytrain, self.Ytest)

		if self.Xtrain is not None and self.smiles_train is None:
			self.kernel.lengthscale, self.kernel.sig_var, self.kernel.noise_var = lat_hyp.compute(Ytrain, Ytest, Xtrain=self.Xtrain, Xtest=self.Xtest)

		elif self.Xtrain is None and self.smiles_train is not None:
			self.kernel.lengthscale, self.kernel.sig_var, self.kernel.noise_var = lat_hyp.compute(Ytrain, Ytest, smiles_train=self.smiles_train, smiles_test=self.smiles_test)

	def regression(self):
		import regression

		self.Ytrain_mean = np.mean(self.Ytrain)
		c_threshold = self.threshold - self.Ytrain_mean
		Ytrain = utils.centre(self.Ytrain)

		Ytest = utils.centre(self.Ytrain, self.Ytest)

		if self.Xtrain is None:
			regress = regression.Regression(Ytrain, Ytest, smiles_train=self.smiles_train, smiles_test=self.smiles_test, kernel=self.kernel, cent_threshold=c_threshold)

		elif self.smiles_train is None:
			regress = regression.Regression(Ytrain, Ytest, Xtrain=self.Xtrain, Xtest=self.Xtest, kernel=self.kernel, cent_threshold=c_threshold)

		else:
			regress = regression.Regression(Ytrain, Ytest, Xtrain=self.Xtrain, Xtest=self.Xtest, smiles_train=self.smiles_train, smiles_test=self.smiles_test, kernel=self.kernel, cent_threshold=c_threshold)

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
