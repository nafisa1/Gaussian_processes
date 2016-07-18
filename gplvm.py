import math
import kernels
import numpy as np
import GPy
import utils
import regression
from scipy.optimize import minimize

# GPLVM with the RBF kernel 

class GPLVM(object):
	def __init__(self, Y, latent_dim, kernel=None, objective=None):
		self.Y = Y
		self.latent_dim = latent_dim
		self.kernel = kernel
		self.objective = objective

		if kernel is None:
			import warnings
			warnings.warn("Kernel not specified, defaulting to RBF kernel...")
			self.kernel = kernels.RBF()

	def remove_zero_cols(self):
	        # Calculate number of observations, N
 		self.N = self.Y.shape[0]

		# Remove columns of zeros
		zeros = np.zeros((1,self.N))[0]
		transpose = self.Y.T

		non_zero_cols = []
		for column in transpose:
    			if np.array_equal(column, zeros) == False:
        			non_zero_cols.append(column)
        
		non_zero_cols = np.array(non_zero_cols)
		self.Y = non_zero_cols.T
		
		return self.Y

	def opt_sep(self):
		X = self.X
		N = self.N
		Q = self.latent_dim
		D = self.D
		Y = self.Y
		kern = self.kernel

		def log_lik():
			X = self.X
			kern = self.kernel
			cov = kern.compute_noisy(X,X)
			inv_cov = np.linalg.inv(cov)
			YYt = np.dot(self.Y, self.Y.T)
			log_l = (-0.5*D*N*np.log(2*math.pi))-(0.5*D*np.log(np.linalg.det(cov))) - (0.5*np.matrix.trace(np.dot(inv_cov,YYt)))  
			return -log_l
		
		def opt_X(X):
			X= np.array(X).reshape((-1, Q))
			kern = self.kernel
			cov = kern.compute_noisy(X,X)
			chol = np.linalg.cholesky(cov)
			inv_chol = np.linalg.inv(chol)
			inv_cov = np.dot(inv_chol.T, inv_chol)
			YYt = np.dot(self.Y, self.Y.T)
			log_l = (-0.5*D*N*np.log(2*math.pi))-(0.5*D*np.log(np.linalg.det(cov))) - (0.5*np.matrix.trace(np.dot(inv_cov,YYt)))  
			return -log_l

		def opt_hyp(hyp):
			X = self.X
			kern = kernels.RBF(lengthscale=hyp[0], sig_var=hyp[1], noise_var=hyp[2])
			cov = kern.compute_noisy(X,X)
#			cov = cov + (0.1*self.kernel.noise_var*np.eye(cov.shape[1])) 
			chol = np.linalg.cholesky(cov)
			inv_chol = np.linalg.inv(chol)
			inv_cov = np.dot(inv_chol.T, inv_chol)
			YYt = np.dot(self.Y, self.Y.T)
			det = np.linalg.det(cov)
#			if det == 0:
#				det = 0.1
#			print hyp
			log_l = (-0.5*self.D*250*np.log(2*math.pi))-(0.5*D*np.log(det)) - (0.5*np.matrix.trace(np.dot(inv_cov,YYt)))  
			return -log_l

#		print log_lik()
		ll_old = log_lik()

		tol = 0.5
		diff = 1

		print "Optimize latent space then hyperparameters, iterate:"
#		while diff > tol:
		for i in xrange(1):
			min_X = minimize(opt_X, self.X.flat, method='l-bfgs-b', options={'disp':True})
			self.X = np.reshape(min_X.x, (N,Q))
#			min_hyp = minimize(opt_hyp, [self.kernel.lengthscale, self.kernel.sig_var, self.kernel.noise_var], method='l-bfgs-b', bounds=((0.5,3),(0.5,3),(0.5,3)), options={'disp':True})

	#		print log_lik()
#			diff = ll_old - log_lik()
#			ll_old = log_lik()

#		self.kernel.lengthscale, self.kernel.sig_var, self.kernel.noise_var = min_hyp.x[0], min_hyp.x[1], min_hyp.x[2]

		return self.X, min_hyp.x

