import numpy as np
import plotting
from regression import Regression
from scipy.stats import norm

class PI(object):

	def compute(self, smiles_test, smiles_train, Ytrain, kern, plot=False):
		# Get posterior mean and standard deviation for test set
		run = Regression(smiles_test=smiles_test, smiles_train=smiles_train, Ytrain=Ytrain, add_noise=0.01, kernel=kern, Ytest=None)
		sd = run.post_s
		p_mean = run.post_mean

		# Find the minimum of the training values
		f_max = np.amax(Ytrain)

		# Calculate input for cdf
		acq1 = (p_mean - f_max)/sd

		# Centre and normalize cdf input so it has zero mean and unit standard deviation
		acq_mean = np.mean(acq1)
		acq_sd = np.std(acq1)
		acq1 = (acq1-acq_mean)/acq_sd

		# Calculate acquisition function
		acq = norm.cdf(acq1)

		# Find maximum of acquisition function and corresponding test input
		ind = np.argmax(acq)
		new_x = smiles_test[ind]
		
		# Take first principal component as X axis for plotting
		#Xtest_axis = Xtest[:,0]	
#		Xtrain_axis = Xtrain[:,0].reshape(-1,1)

		# Plot posterior and acquisition function, showing preferred next observation
		#if plot==True:		
		#	plotting.plot_acq(Xtest_axis, acq, p_mean, sd, Ytest=None)

		return new_x, ind
		
class EI(object):

	def compute(self, smiles_test, smiles_train, Ytrain, kern, plot=False):
		# Get posterior mean and standard deviation for test set
		run = Regression(smiles_test=smiles_test, smiles_train=smiles_train, Ytrain=Ytrain, add_noise=0.01, kernel=kern, Ytest=None)
		sd = run.post_s
		p_mean = run.post_mean

		# Find the maximum of the training values
		f_max = np.amax(Ytrain)

		# Calculate input for cdf
		acq1 = (p_mean - f_max)/sd

		# Centre and normalize cdf input so it has zero mean and unit standard deviation
		acq_mean = np.mean(acq1)
		acq_sd = np.std(acq1)
		acq1 = (acq1-acq_mean)/acq_sd

		# Calculate acquisition function
		acq = sd*((acq1*norm.cdf(acq1)) + norm.pdf(acq1))

		# Find maximum of acquisition function and corresponding test input
		ind = np.argmax(acq)
		new_x = smiles_test[ind]
		
		# Take first principal component as X axis for plotting
#		Xtest_axis = Xtest[:,0]	
#		Xtrain_axis = Xtrain[:,0].reshape(-1,1)

#		if plot==True:
		# Plot posterior and acquisition function, showing preferred next observation
#			plotting.plot_acq(Xtest_axis, acq, p_mean, sd, Ytest=None)

		return new_x, ind

class UCB(object):

	def __init__(self, kappa=0.2):
		self.kappa = kappa

	def compute(self, smiles_test, smiles_train, Ytrain, kern, plot=False):
		# Get posterior mean and standard deviation for test set
		run = Regression(smiles_test=smiles_test, smiles_train=smiles_train, Ytrain=Ytrain, add_noise=0.01, kernel=kern, Ytest=None)
		sd = run.post_s
		p_mean = run.post_mean

		# Calculate acquisition function
		acq = p_mean + (self.kappa*sd)

		# Find maximum of acquisition function and corresponding test input
		ind = np.argmax(acq)
		new_x = smiles_test[ind]
		print max(acq)
		
		# Take first principal component as X axis for plotting
		#Xtest_axis = Xtest[:,0]	
#		Xtrain_axis = Xtrain[:,0].reshape(-1,1)

		#if plot==True:
		# Plot posterior and acquisition function, showing preferred next observation
		#	plotting.plot_acq(Xtest_axis, acq, p_mean, sd, Ytest=None)

		return new_x, ind

class LCB(object):

	def compute(self, Xtest, Xtrain, Ytrain, kern, kappa=0.2, plot=False):
		# Get posterior mean and standard deviation for test set
		run = Regression(Xtest, Xtrain, Ytrain, add_noise=0.01, kernel=kern, Ytest=None)
		sd = run.post_s
		p_mean = run.post_mean

		# Calculate acquisition function
		acq = p_mean - (kappa*sd)

		# Find maximum of acquisition function and corresponding test input
		ind = np.argmin(acq)
		new_x = Xtest[ind]
		
		# Take first principal component as X axis for plotting
		Xtest_axis = Xtest[:,0]	
#		Xtrain_axis = Xtrain[:,0].reshape(-1,1)

		if plot==True:
		# Plot posterior and acquisition function, showing preferred next observation
			plotting.plot_acq(Xtest_axis, acq, p_mean, sd, Ytest=None)

		return new_x, ind


