import numpy as np

# Latin hypercube sampling

class LHS(object):
	def __init__(self, parameters=3, divisions=6, lower=0.5, upper=2.5):
		self.parameters = parameters
		self.divisions = divisions
		self.lower = lower
		self.upper = upper

		a = np.linspace(lower,upper,divisions)

		div_list = []

		for i in xrange(parameters):
			div_list.append(np.random.permutation(a))
	
		div_list = np.array((div_list))

		combinations = np.dstack(div_list)
		self.combinations = combinations[0]

	
