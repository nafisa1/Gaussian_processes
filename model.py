import numpy as np
import utils
import GPy
import kernels
import cross_validation # CIRCULAR IMPORT
import max_likelihood
import regression

class Model(object):

	def __init__(self, n_kers=1, pca=False, print_jit=False, latent_dim=None, X=None, Y=None, Ytrain=None, Ytest=None, Xtrain=None, Xtest=None, kernel=None, prior_train=None, prior_test=None, acq_func=None, threshold=None):

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

		if prior_train is not None:
			prior_train = prior_train.reshape(-1,1)
			# subtract training prior mean from Ytrain values
			self.Ytrain = self.Ytrain - prior_train
		
		self.hparameter_choices = utils.LHS(parameters=n_kers*2)






	def hyperparameters(self, frac_test=0.2, num_folds=10, max_ll=True, print_vals=True, split=False):
		if split == True:
			cross_val = cross_validation.Cross_Validation(self.X, self.Y, fraction_test=frac_test, n_folds=num_folds, n_kers=self.n_kers, threshold=self.threshold)
			cross_val.order()
			self.Xtest, self.Ytest, self.Xtrain, self.Ytrain = cross_val.get_test_set()
			best_noise_var, all_means, iteration_means = cross_val.repeated_CV(self.kernel, self.hparameter_choices, iterations=1, lhs_kern=self.kernel) # EDIT BACK TO 10 ITERATIONS
			self.kernel.noise_var = best_noise_var
		if max_ll == True:
			best_hparams = self.max_log_likelihood(print_vals=print_vals)
		
		print best_hparams

	def max_log_likelihood(self, print_vals=True):

		final_points = []
		log_likelihoods = []
		centred_Ytrain = utils.centre(self.Ytrain.reshape(-1,1))

		find_max_ll = max_likelihood.Max_LL(centred_Ytrain, self.kernel)

		default_starting_point = []
		
		if isinstance(self.kernel, kernels.Composite):
			for item in self.kernel.kers:
				if item is not None:
					default_starting_point.append(item.lengthscale)
					default_starting_point.append(item.sig_var)
		else:
			default_starting_point.append(self.kernel.lengthscale)
			default_starting_point.append(self.kernel.sig_var)
		final_point, ll = find_max_ll.run_opt(default_starting_point)
		if print_vals==True:
			print final_point, ll
		final_points.append(final_point)
		log_likelihoods.append(ll)

		for choice in self.hparameter_choices:
			starting_point = []
			if isinstance(self.kernel, kernels.Composite):
				for item in self.kernel.kers:
					if item is not None:
						starting_point.append(choice[0])
						starting_point.append(choice[1]) # check this is correct
			else:
				starting_point.append(choice[0])
				starting_point.append(choice[1])
			final_point, ll = find_max_ll.run_opt(starting_point)

			if print_vals==True:
				print final_point, ll
			final_points.append(final_point)
			log_likelihoods.append(ll)
		index = np.argmin(log_likelihoods)
		best_hparams = final_points[index]
		if print_vals==True:
			print "Best hyperparameters:", best_hparams

		if isinstance(self.kernel, kernels.Composite):
			count = 0
			for item in self.kernel.kers:
				if item is not None:
					item.lengthscale = best_hparams[count]
					item.sig_var = best_hparams[count+1]
					count +=2
		else:
			self.kernel.lengthscale = best_hparams[0]
			self.kernel.sig_var = best_hparams[1]

		return best_hparams

	def regression(self):


		self.Ytrain_mean = np.mean(self.Ytrain)
		if self.threshold is not None:
			c_threshold = self.threshold - self.Ytrain_mean
		else:
			c_threshold = None
		Ytrain = utils.centre(self.Ytrain)


		Ytest = utils.centre(self.Ytrain, self.Ytest)

		regress = regression.Regression(Ytrain, Ytest=Ytest, Xtrain=self.Xtrain, Xtest=self.Xtest, kernel=self.kernel, cent_threshold=c_threshold)

		return regress

	def optimization(self, plot=False):

		Ytrain = utils.centre(self.Ytrain)

# CENTRE XTRAIN AGAIN?

		if plot==False:
			new_x, ind = self.acq_func.compute(self.Xtest, self.Xtrain, Ytrain, self.kernel, plot=False)
		else:
			new_x, ind = self.acq_func.compute(self.Xtest, self.Xtrain, Ytrain, self.kernel, plot=True)

		new_obs = self.Ytest[ind]
#		print new_x	
		if len(self.Xtrain) != 2:
			if isinstance(new_x, float) == True:
				self.Xtrain = np.vstack((self.Xtrain, new_x))
				self.Xtest = np.delete(self.Xtest, ind, axis=0)
			else:
				self.Xtrain.append(new_x)
				del self.Xtest[ind]
		else:
			self.Xtrain[0] = np.vstack((self.Xtrain[0], new_x[0]))
			self.Xtrain[1].append(new_x[1])
			self.Xtest[0] = np.delete(self.Xtest[0], ind, axis=0)
			del self.Xtest[1][ind]
#		self.Xtrain = np.vstack((self.Xtrain, new_x))
		self.Ytrain = np.vstack((self.Ytrain, new_obs))

#		self.Xtest = np.delete(self.Xtest, ind, axis=0)
		self.Ytest = np.delete(self.Ytest, ind, axis=0)

		return new_x

	# This is done after regression etc			
	def correction(self):
		assert prior is not None, "Posterior mean correction is not required for a zero mean prior."
		# add test prior mean to posterior mean
		self.Ytest = self.Ytest + self.prior_test
		# will also have to apply correction to random draws
