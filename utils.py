import numpy as np

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

	
