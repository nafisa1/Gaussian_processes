import math
import numpy as np
import kernels
import utils
from scipy.optimize import minimize 

class GPLVM(object):
	def __init__(self, X, kernel, N, D):
		self.X = X
		self.kernel = kernel
		self.N = N
		self.D = D

	def log_lik(Y, kernel):
		cov = kernel.compute(Y,Y, noise=False) # noise should be false but add jitter if returned by jitchol
		chol_cov, jitter = kernels.jit_chol(cov, attempts=100,print_jit=False)
		if jitter != 0:
		    kernel.noise_var = jitter
        	cov = kernel.compute(Y,Y, noise=True)
		Yt_invcov_Y = np.dot(np.linalg.solve(chol_cov, Y).T, np.linalg.solve(chol_cov, Y))
		if np.linalg.det(cov) == 0:
			determinant = 3.0e-324
		else:
			determinant = np.linalg.det(cov)
		log_l = (-0.5*Y.shape[0]*np.log(2*math.pi))-(0.5*np.log(determinant)) - (0.5*Yt_invcov_Y)
		return log_l

	def opt_hyp(hyp):
		k.lengthscale=hyp[0]
		k.sig_var=hyp[1]
		log_l = log_lik(c_output, k)
		return -log_l

	def opt_hyp_composite(hyp):
		if isinstance(self.kernel, kernels.Composite):
			count = 0
			for i in xrange(self.kernel.nkers):
				for item in self.kernel.kers:
					item.lengthscale = hyp[count]
					item.sig_var = hyp[count+1]
					item.noise_var = hyp[count+2]
					count +=3
			kern = self.kernel
		else:			
			kern = self.kernel(lengthscale=hyp[0], sig_var=hyp[1], noise_var=hyp[2])
		#	k.lengthscale=hyp[0]
		#	k.sig_var=hyp[1]
		#	k.noise_var=hyp[2]
		log_l = log_lik(outputs, k)
		return -log_l
		
	diff = 1.0
	tol = 0.5

	log_likelihoods = [] # save for plotting / get function values from optimizer?
	print "Optimize hyperparameters:"
	for i in xrange(5):#while diff > tol:

		if isinstance(self.kernel, kernels.Composite):
			hyp = []
			for i in xrange(self.kernel.nkers):
				for item in self.kers:
					hyp.append(item.lengthscale)
					hyp.append(item.sig_var)
					hyp.append(item.noise_var)
		else:
			hyp = [self.kernel.lengthscale, self.kernel.sig_var, self.kernel.noise_var]
			min_hyp = minimize(opt_hyp, [self.kernel.lengthscale, self.kernel.sig_var, self.kernel.noise_var], method='l-bfgs-b', bounds=((0.2,3),(0.2,3),(0.2,3)), options={'disp':True})

		print -log_lik()
		print diff, tol
		log_likelihoods.append(-log_lik())
		diff = ll_old - log_lik()
		ll_old = log_lik()

	self.kernel.lengthscale, self.kernel.sig_var, self.kernel.noise_var = min_hyp.x[0], min_hyp.x[1], min_hyp.x[2]

	return min_hyp.x, log_likelihoods

