import math
import kernels
import numpy as np
import GPy
import utils
import regression_2D
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

	def initialize(self):
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

		# Normalize
		mu = np.vstack(np.mean(self.Y, axis=0))
		s = np.vstack(self.Y.std(axis=0))
		centred = self.Y.T - mu
		div = centred/s                       
		self.Y = div.T 

		# Calculate number of observed dimensions, D
		self.D = self.Y.shape[1]

		# Initialize latent points using PCA
		self.X,W = GPy.util.linalg.pca(self.Y, self.latent_dim)
		jitter = 0.05*np.random.rand((self.X.shape[0]), (self.X.shape[1]))
		jitter -= 0.025
		self.X -= jitter
		print self.D,"observed dimensions,",self.latent_dim,"latent dimensions."
		return self.X

	def lhs(self, print_output=True):
		lat_hyp = utils.LHS()
		comb = lat_hyp.combinations

		X_train = self.X[:int((self.X.shape[0])*0.95),:]
		X_test = self.X[int((self.X.shape[0])*0.95):,:]
		Y_train = self.objective[:int((self.objective.shape[0])*0.95),:]
		Y_test = self.objective[int((self.objective.shape[0])*0.95):,:]

		r_sq = []
		kern = kernels.RBF()

		for i in xrange(lat_hyp.divisions):
			kern.lengthscale = comb[i][0]
			kern.sig_var = comb[i][1]
			kern.noise_var = comb[i][2]
			regr = regression_2D.Regression(X_test, X_train, Y_train, add_noise=0, kernel=kern)
			r_sq.append(regr.r_squared(Y_test))

		ind = r_sq.index(max(r_sq))
		best = comb[ind]
		self.kernel.lengthscale = best[0]
		self.kernel.sig_var = best[1]
		self.kernel.noise_var = best[2]

		if print_output is True:
			print "The new kernel hyperparameters are: lengthscale=",self.kernel.lengthscale,", power=",self.kernel.sig_var," and noise variance=",self.kernel.noise_var,"."

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
			inv_cov = np.linalg.inv(cov)
			YYt = np.dot(self.Y, self.Y.T)
			log_l = (-0.5*D*N*np.log(2*math.pi))-(0.5*D*np.log(np.linalg.det(cov))) - (0.5*np.matrix.trace(np.dot(inv_cov,YYt)))  
			return -log_l

		def opt_hyp(hyp):
			X = self.X#[:250,:]
			kern = kernels.RBF(lengthscale=hyp[0], sig_var=hyp[1], noise_var=hyp[2])
			cov = kern.compute_noisy(X,X)
			cov = cov + (0.1*self.kernel.noise_var*np.eye(cov.shape[1])) 
			chol = np.linalg.cholesky(cov)
			chol_inv = np.linalg.inv(chol)
			#cove = cov + (0.01*self.kernel.noise_var*np.eye(cov.shape[1]))
			inv_cov = np.dot(chol_inv.T, chol_inv)
			YYt = np.dot(self.Y, self.Y.T)
			c_cov = np.dot(chol, chol.T)
			det = np.linalg.det(c_cov)
			if det == 0:
				det = 0.1
			print hyp
			log_l = (-0.5*self.D*250*np.log(2*math.pi))-(0.5*D*np.log(det)) - (0.5*np.matrix.trace(np.dot(inv_cov,YYt)))  
			return -log_l

#		print log_lik()
		ll_old = log_lik()

		tol = 0.5
		diff = 1

		print "Optimize latent space then hyperparameters, iterate:"
		while diff > tol:
#			min_X = minimize(opt_X, self.X.flat, method='l-bfgs-b', options={'disp':True})
#			self.X = np.reshape(min_X.x, (N,Q))
			min_hyp = minimize(opt_hyp, [self.kernel.lengthscale, self.kernel.sig_var, self.kernel.noise_var], method='l-bfgs-b', bounds=((0,None),(0,None),(0,None)), options={'disp':True})
	#		print log_lik()
			diff = ll_old - log_lik()
			ll_old = log_lik()
		self.kernel.lengthscale, self.kernel.sig_var, self.kernel.noise_var = min_hyp.x[0], min_hyp.x[1], min_hyp.x[2]

		return self.X, min_hyp.x

