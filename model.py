import numpy as np
import utils
import GPy
import max_likelihood
import regression

class Model(object):

  def __init__(self, n_kers=2, composite_ker=True, print_jit=False, Y=None, Ytrain=None, Ytest=None, Xtrain=None, Xtest=None, kernel=None, acq_func=None, threshold=None):

		self.n_kers = n_kers # Update kernel module so number of kernels is given
		self.composite_ker = composite_ker
		self.print_jit = print_jit
		self.Xtrain = Xtrain
		self.Xtest = Xtest
		self.Ytrain = Ytrain
		self.Ytest = Ytest
		self.kernel = kernel
		self.acq_func = acq_func
		self.threshold = threshold

		if len(Ytrain.shape) != 2:
			self.Ytrain = Ytrain.reshape(-1,1)	
		if Ytest is not None:
			if len(Ytest.shape) != 2:
				self.Ytest = Ytest.reshape(-1,1)
		
		self.hparameter_choices = utils.LHS(parameters=n_kers*2)

  def hyperparameters(self, max_ll=True, print_vals=True):
		if max_ll == True:
			best_hparams = self.max_log_likelihood(print_vals=print_vals)
		
		if print_vals == True:
			print best_hparams

  def max_log_likelihood(self, print_vals=True):

		final_points = []
		log_likelihoods = []
		jitters = []
		centred_Ytrain = utils.centre(self.Ytrain.reshape(-1,1))

		find_max_ll = max_likelihood.Max_LL(centred_Ytrain, self.kernel, self.print_jit)
		default_starting_point = []		
		if self.composite_ker==True: 
			for item in self.kernel.kers:
				if item is not None:
					default_starting_point.append(item.lengthscale)
					default_starting_point.append(item.sig_var)
		else:
			default_starting_point.append(self.kernel.lengthscale)
			default_starting_point.append(self.kernel.sig_var)
		final_point, ll, jitter = find_max_ll.run_opt(default_starting_point)
		jitters.append(jitter)
		if print_vals==True:
			print final_point, ll
		final_points.append(final_point)
		log_likelihoods.append(ll)
		for choice in self.hparameter_choices:
			starting_point = []
			if self.composite_ker==True: 
				for item in self.kernel.kers:
					if item is not None:
						starting_point.append(choice[0])
						starting_point.append(choice[1]) # check this is correct
			else:
				starting_point.append(choice[0])
				starting_point.append(choice[1])
			final_point, ll, jitter = find_max_ll.run_opt(starting_point)
			jitters.append(jitter)
			if print_vals==True:
				print final_point, ll
			final_points.append(final_point)
			log_likelihoods.append(ll)
		index = np.argmin(log_likelihoods)
		best_hparams = final_points[index]
		best_jitter = jitters[index]
		self.kernel.noise_var += best_jitter
		if print_vals==True:
			print "Best hyperparameters:", best_hparams
			print "Jitter", best_jitter
			print "Noise variance =", self.kernel.noise_var

		if self.composite_ker==True: 
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

  def optimization(self):

    Ytrain = utils.centre(self.Ytrain)
		
    run = regression.Regression(Xtest=self.Xtest, Xtrain=self.Xtrain, Ytrain=Ytrain, add_noise=0.0, kernel=self.kernel, Ytest=None)
    sd = run.post_s
    p_mean = run.post_mean
# CENTRE XTRAIN AGAIN?

    new_x, ind = self.acq_func.compute(self.Xtest, self.Xtrain, Ytrain, sd, p_mean, plot=False)

    new_obs = self.Ytest[ind]

    if len(self.Xtrain) != 2:
      if isinstance(new_x, float) == True:
        self.Xtrain = np.vstack((self.Xtrain, new_x))
        self.Xtest = np.delete(self.Xtest, ind, axis=0)
      else:
        self.Xtrain.append(new_x)
        del self.Xtest[ind]
    else:
      self.Xtrain[0] = np.vstack((self.Xtrain[0], new_x[0]))
      self.Xtrain[1] = list(self.Xtrain[1])
      self.Xtrain[1].append(new_x[1])
      self.Xtest[0] = np.delete(self.Xtest[0], ind, axis=0)
      self.Xtest[1] = list(self.Xtest[1])
      del self.Xtest[1][ind]
      
    self.Ytrain = np.vstack((self.Ytrain, new_obs))
    self.Ytest = np.delete(self.Ytest, ind, axis=0)
    
    return new_x, new_obs

# subtract training prior mean from Ytrain values
# Correction - done after regression etc			
# add test prior mean to posterior mean
# will also have to apply correction to random draws
