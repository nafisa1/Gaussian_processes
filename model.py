import numpy as np
import utils
import GPy
import kernels
import cross_validation # CIRCULAR IMPORT
import max_likelihood

class Model(object):

	def __init__(self, n_kers=1, pca=False, print_jit=False, latent_dim=None, X=None, Y=None, Ytrain=None, Ytest=None, Xtrain=None, Xtest=None, smiles_train=None, smiles_test=None, kernel=None, prior_train=None, prior_test=None, acq_func=None, threshold=None):

		self.n_kers = n_kers # Update kernel module so number of kernels is given
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
		
		self.hparameter_choices = utils.LHS().combinations

	def hyperparameters(self, frac_test=0.2, num_folds=10, max_ll=True, split=False, ll_kernel=None):
		if split == True:
			cross_val = cross_validation.Cross_Validation(self.X, self.Y, fraction_test=frac_test, n_folds=num_folds, n_kers=self.n_kers, threshold=self.threshold)
			cross_val.order()
			self.smiles_test, self.Ytest, self.smiles_train, self.Ytrain = cross_val.get_test_set()
			best_noise_var, all_means, iteration_means = cross_val.repeated_CV(self.kernel, self.hparameter_choices, iterations=1, lhs_kern=self.kernel) # EDIT BACK TO 10 ITERATIONS
			self.kernel.noise_var = best_noise_var
		if max_ll == True:
			self.max_log_likelihood(opt_kernel=ll_kernel)
		
		print "The kernel hyperparameters are: lengthscale", self.kernel.lengthscale,"signal variance", self.kernel.sig_var,"noise variance", self.kernel.noise_var,"."

	def max_log_likelihood(self, opt_kernel=None):
		final_points = []
		log_likelihoods = []
		centred_Ytrain = utils.centre(self.Ytrain.reshape(-1,1))
		if opt_kernel is None:
			find_max_ll = max_likelihood.Max_LL(centred_Ytrain, self.kernel)
		else:
			find_max_ll = max_likelihood.Max_LL(centred_Ytrain, opt_kernel)

		default_starting_point = []
		default_starting_point.append(self.kernel.lengthscale)
		default_starting_point.append(self.kernel.sig_var)
		final_point, ll = find_max_ll.run_opt(default_starting_point)
		print final_point, ll
		final_points.append(final_point)
		log_likelihoods.append(ll)

		for choice in self.hparameter_choices:
			starting_point = []
			starting_point.append(choice[0])
			starting_point.append(choice[1])
			final_point, ll = find_max_ll.run_opt(starting_point)
			print final_point, ll
			final_points.append(final_point)
			log_likelihoods.append(ll)
		index = np.argmin(log_likelihoods)
		best_hparams = final_points[index]
		print "Best hyperparameters:", best_hparams
		self.kernel.lengthscale = best_hparams[0]
		self.kernel.sig_var = best_hparams[1]

	def regression(self):
		import regression

		self.Ytrain_mean = np.mean(self.Ytrain)
		if self.threshold is not None:
			c_threshold = self.threshold - self.Ytrain_mean
		else:
			c_threshold = None
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
			new_x, ind = self.acq_func.compute(self.smiles_test, self.smiles_train, Ytrain, self.kernel, plot=False)
		else:
			new_x, ind = self.acq_func.compute(self.smiles_test, self.smiles_train, Ytrain, self.kernel, plot=True)

		new_obs = self.Ytest[ind]		
		print len(self.smiles_train)
		print len(new_x)
		print self.Ytrain.shape
		print new_obs.shape
		self.smiles_train.append(new_x)
#		self.smiles_train = np.vstack((self.smiles_train, new_x))
		self.Ytrain = np.vstack((self.Ytrain, new_obs))

		del self.smiles_test[ind]
#		self.smiles_test = np.delete(self.smiles_test, ind, axis=0)
		self.Ytest = np.delete(self.Ytest, ind, axis=0)

		return new_x

	# This is done after regression etc			
	def correction(self):
		assert prior is not None, "Posterior mean correction is not required for a zero mean prior."
		# add test prior mean to posterior mean
		self.Ytest = self.Ytest + self.prior_test
		# will also have to apply correction to random draws
