import math
import numpy as np
from scipy.optimize import minimize

# GPLVM with the RBF kernel 

class GPLVM(object):
	def __init__(self, X, Y, kernel, dim=1):
		self.X = X
		self.Y = Y
		self.kernel = kernel
		self.dim = dim

	def opt_sep(self):
		X = self.X
		N = 1000
		D = self.dim
#		Y = self.Y
		kern = self.kernel

		def log_lik():
			X = self.X
			kern = self.kernel
			cov = kern.compute_noisy(X,X)
			inv_cov = np.linalg.inv(cov)
			YYt = np.dot(self.X, self.X.T)
			log_l = (-0.5*D*N*np.log(2*math.pi))-(0.5*D*np.log(np.linalg.det(cov))) - (0.5*np.matrix.trace(np.dot(inv_cov,YYt)))  
			return -log_l
		
#		def opt_X(X):
#			X= np.array(X).reshape((-1, Q))
#			kern = self.kernel
#			cov = kern.compute_noisy(X,X)
#			chol = np.linalg.cholesky(cov)
#			inv_chol = np.linalg.inv(chol)
#			inv_cov = np.dot(inv_chol.T, inv_chol)
#			YYt = np.dot(self.Y, self.Y.T)
#			log_l = (-0.5*D*N*np.log(2*math.pi))-(0.5*D*np.log(np.linalg.det(cov))) - (0.5*np.matrix.trace(np.dot(inv_cov,YYt)))  
#			return -log_l

		def opt_hyp(hyp):
			X = self.X
			y = self.Y
			kern = self.kernel(lengthscale=hyp[0], sig_var=hyp[1], noise_var=hyp[2])
			cov = kern.compute_noisy(X,X)
			chol = np.linalg.cholesky(cov)
			inv_chol = np.linalg.inv(chol)
			inv_cov = np.dot(inv_chol.T, inv_chol)
			YYt = np.dot(self.X, self.X.T)
			det = np.linalg.det(cov)
#			if det == 0:
#				det = 0.1
#			print hyp
			log_l = (-0.5*D*250*np.log(2*math.pi))-(0.5*D*np.log(det)) - (0.5*np.matrix.trace(np.dot(inv_cov,YYt)))  
			return -log_l

#		print log_lik()
		ll_old = log_lik()

		tol = 0.5
		diff = 1

		print "Optimize hyperparameters:"
		while diff > tol:
#			min_X = minimize(opt_X, self.X.flat, method='l-bfgs-b', options={'disp':True})
#			self.X = np.reshape(min_X.x, (N,Q))
			min_hyp = minimize(opt_hyp, [self.kernel.lengthscale, self.kernel.sig_var, self.kernel.noise_var], method='l-bfgs-b', bounds=((0.2,3),(0.2,3),(0.2,3)), options={'disp':True})

			print log_lik()
			diff = ll_old - log_lik()
			ll_old = log_lik()

		self.kernel.lengthscale, self.kernel.sig_var, self.kernel.noise_var = min_hyp.x[0], min_hyp.x[1], min_hyp.x[2]

		return min_hyp.x

