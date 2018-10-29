import numpy as np
from scipy.stats import norm

class Random(object):
	def __init__(self, abbreviation="rand"):
		self.abbreviation = abbreviation

	def compute(self, Xtest, Xtrain, Ytrain, sd, p_mean, plot=False):

		if len(Xtest) != 2:
			ind = np.random.randint(len(Xtest))
			new_x = Xtest[ind]
		else:
			ind = np.random.randint(len(Xtest[0]))
			new_x = []
			new_x.append(Xtest[0][ind]) # new numerical value(s)
			new_x.append(Xtest[1][ind]) # new smiles
		
		return new_x, ind

class Chronological(object):
	def __init__(self, abbreviation="chron"):
		self.abbreviation = abbreviation

	def compute(self, Xtest, Xtrain, Ytrain, sd, p_mean, plot=False):
		ind = 0
		if len(Xtest) != 2:
			new_x = Xtest[ind]
		else:
			new_x = []
			new_x.append(Xtest[0][ind]) # new numerical value(s)
			new_x.append(Xtest[1][ind]) # new smiles
		
		return new_x, ind

class Unc(object):
	def __init__(self, abbreviation="unc"):
		self.abbreviation = abbreviation

	def compute(self, Xtest, Xtrain, Ytrain, sd, p_mean, plot=False):
		# Find maximum of acquisition function and corresponding test input
		ind = np.argmax(sd)
		if len(Xtest) != 2:
			new_x = Xtest[ind]
		else:
			new_x = []
			new_x.append(Xtest[0][ind]) # new numerical value(s)
			new_x.append(Xtest[1][ind]) # new smiles
		
		return new_x, ind

class PI(object):
	def __init__(self, eta=0.0, abbreviation="pi"):
		self.eta = eta
		self.abbreviation = abbreviation

	def compute(self, Xtest, Xtrain, Ytrain, sd, p_mean, plot=False):
		# sd p_mean 

		# Find the minimum of the training values
		f_max = np.amax(Ytrain)

		# Calculate input for cdf
		acq1 = (p_mean - f_max - self.eta)/sd

		# Centre and normalize cdf input so it has zero mean and unit standard deviation
		acq_mean = np.mean(acq1)
		acq_sd = np.std(acq1)
		acq1 = (acq1-acq_mean)/acq_sd

		# Calculate acquisition function
		acq = norm.cdf(acq1)

		# Find maximum of acquisition function and corresponding test input
		ind = np.argmax(acq)
		if len(Xtest) != 2:
			new_x = Xtest[ind]
		else:
			new_x = []
			new_x.append(Xtest[0][ind]) # new numerical value(s)
			new_x.append(Xtest[1][ind]) # new smiles

		return new_x, ind
		
class EI(object):
	def __init__(self, eta=0.0, abbreviation="ei"):
		self.eta = eta
		self.abbreviation = abbreviation

	def compute(self, Xtest, Xtrain, Ytrain, sd, p_mean, plot=False):
		# sd and p_mean

		# Find the maximum of the training values
		f_max = np.amax(Ytrain)

		# Calculate input for cdf
		acq1 = (p_mean - f_max - self.eta)/sd

		# Centre and normalize cdf input so it has zero mean and unit standard deviation
		acq_mean = np.mean(acq1)
		acq_sd = np.std(acq1)
		acq1 = (acq1-acq_mean)/acq_sd

		# Calculate acquisition function
		acq = sd*((acq1*norm.cdf(acq1)) + norm.pdf(acq1))

		# Find maximum of acquisition function and corresponding test input
		ind = np.argmax(acq)
		if len(Xtest) != 2:
			new_x = Xtest[ind]
		else:
			new_x = []
			new_x.append(Xtest[0][ind]) # new numerical value(s)
			new_x.append(Xtest[1][ind]) # new smiles

		return new_x, ind

class UCB(object):
	def __init__(self, kappa=0.2, abbreviation="ucb"):
		self.kappa = kappa
		self.abbreviation = abbreviation

	def compute(self, Xtest, Xtrain, Ytrain, sd, p_mean, plot=False):
		# sd and p_mean
		# Calculate acquisition function
		acq = p_mean + (self.kappa*sd)

		# Find maximum of acquisition function and corresponding test input
		ind = np.argmax(acq)
		if len(Xtest) != 2:
			new_x = Xtest[ind]
		else:
			new_x = []
			new_x.append(Xtest[0][ind]) # new numerical value(s)
			new_x.append(Xtest[1][ind]) # new smiles

		return new_x, ind

class LCB(object):

	def __init__(self, kappa=0.2, abbreviation="lcb"):
		self.kappa = kappa
		self.abbreviation = abbreviation

	def compute(self, Xtest, Xtrain, Ytrain, sd, p_mean, plot=False):
		# sd and p_mean

		# Calculate acquisition function
		acq = p_mean - (self.kappa*sd)

		# Find maximum of acquisition function and corresponding test input
		ind = np.argmin(acq)
		if len(Xtest) != 2:
			new_x = Xtest[ind]
		else:
			new_x = []
			new_x.append(Xtest[0][ind]) # new numerical value(s)
			new_x.append(Xtest[1][ind]) # new smiles

		return new_x, ind


