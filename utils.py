import numpy as np

def normalize_centre(A, B=None):
	# Get mean and standard deviation
	A_mu = np.vstack(np.mean(A, axis=0))
	A_sd = np.vstack(A.std(axis=0))

	# Centre and normalize array
	if B is None:
		A_centred = A.T - A_mu
		Acent_normalized = A_centred/A_sd                       
		final_array = Acent_normalized.T   

	# Centre and normalize second array using mean and standard deviation of first array
	else:
		B_centred = B.T - A_mu
		Bcent_normalized = B_centred/A_sd
		final_array = Bcent_normalized.T

	return final_array		

def centre(A, B=None):
	# Get mean
	A_mu = np.vstack(np.mean(A, axis=0))

	# Centre array
	if B is None:
		A_centred = A.T - A_mu                    
		centred_array = A_centred.T   

	# Centre second array using mean of first array
	else:
		B_centred = B.T - A_mu
		centred_array = B_centred.T

	return centred_array	

# Latin hypercube sampling

class LHS(object):
	def __init__(self, parameters=3, n_choices=15, lower=0.5, upper=2.5, divisions=6):
		self.parameters = parameters
		self.divisions = divisions
		self.lower = lower
		self.upper = upper

		a = np.linspace(lower,upper,divisions)
		options = len(a)

		full = np.zeros(((options**3),parameters))

		full[:,0] = np.concatenate(((options**2)*[a[0]], (options**2)*[a[1]], (options**2)*[a[2]], (options**2)*[a[3]], (options**2)*[a[4]], (options**2)*[a[5]]))

		full[:,1] = np.concatenate(options*[np.concatenate((options*[a[0]], options*[a[1]], options*[a[2]], options*[a[3]], options*[a[4]], options*[a[5]]))])

		full[:,2] = np.concatenate((options**2)*[a])

		self.combinations = full[np.random.randint(full.shape[0], size=n_choices),:]

	
