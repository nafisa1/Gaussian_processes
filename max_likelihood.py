import math
import numpy as np
import kernels
import utils
from scipy.optimize import minimize 

class Max_LL(object):
	def __init__(self, Y, kernel):
		self.Y = Y
		self.kernel = kernel

	def log_lik(self):
		cov = self.kernel.compute(self.Y,self.Y, noise=False) # noise should be false but add jitter if returned by jitchol
		chol_cov, jitter = kernels.jit_chol(cov, attempts=100,print_jit=False)
		if jitter != 0:
		    self.kernel.noise_var = jitter
        	cov = self.kernel.compute(self.Y,self.Y, noise=True)
		Yt_invcov_Y = np.dot(np.linalg.solve(chol_cov, self.Y).T, np.linalg.solve(chol_cov, self.Y))
		if np.linalg.det(cov) == 0:
			determinant = 3.0e-324
		else:
			determinant = np.linalg.det(cov)
		log_l = (-0.5*self.Y.shape[0]*np.log(2*math.pi))-(0.5*np.log(determinant)) - (0.5*Yt_invcov_Y)
		return log_l

	def opt_hyp(self, hyp):
		self.kernel.lengthscale=hyp[0]
		self.kernel.sig_var=hyp[1]
		log_l = self.log_lik()
		return -log_l

	def opt_hyp_composite(self, hyp):
		if isinstance(self.kernel, kernels.Composite):
			count = 0
			for i in xrange(self.kernel.nkers):
				for item in self.kernel.kers:
					item.lengthscale = hyp[count]
					item.sig_var = hyp[count+1]
					count +=2
		log_l = self.log_lik()
		return -log_l

	def run_opt(self, hyp):
		min_hyp = minimize(self.opt_hyp, hyp, method='l-bfgs-b', bounds=((0.5,10.0),(0.5,7.0)), options={'disp':True}) 
		#log_likelihoods = [] # save for plotting / get function values from optimizer?
		return min_hyp.x, min_hyp.fun


