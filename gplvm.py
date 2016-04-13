import math
import kernels
import numpy as np
import GPy
from scipy.optimize import minimize

# GPLVM with the RBF kernel 

class GPLVM(object):
	def __init__(self, Y, latent_dim, kernel=None):
		self.Y = Y
		self.latent_dim = latent_dim
		self.kernel = kernel

		if kernel is None:
			import warnings
			warnings.warn("Kernel not specified, defaulting to RBF kernel...")
			self.kernel = kernels.RBF()

	def initialize(self, normalize=True):
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

		# Calculate number of observed_dim, D
		self.D = self.Y.shape[1]

		# Initialize latent points using PCA
		self.X,W = GPy.util.linalg.pca(self.Y, self.latent_dim)
		return self.X

	def get_latent(self):
		X = self.X
		N = self.N
		Q = self.latent_dim
		D = self.D
		Y = self.Y
		kern = self.kernel

		def f(X):
			X = X.reshape((N,Q))
			cov = kern.compute_noisy(X,X)
			inv_cov = np.linalg.inv(cov)
			YYt = np.dot(Y, Y.T)
			log_l = (-0.5*D*N*np.log(2*math.pi))-(0.5*D*np.log(np.linalg.det(cov))) - (0.5*np.matrix.trace(np.dot(inv_cov,YYt)))
			return -log_l
    
		def grad(X):
		 	X = X.reshape(N,-1)
			cov = kern.compute_noisy(X,X)
			inv_cov = np.linalg.inv(cov)
			YYt = np.dot(Y, Y.T)
			dlogl_dK = np.dot(np.dot(inv_cov,YYt),inv_cov) - D*inv_cov
			dK_dX = np.empty((X.shape[0], X.shape[0], X.shape[1]))
			Q = int(X.shape[1])
			for j in range(0,X.shape[0]):
				for i in range(0,X.shape[0]):
					for k in range(0,X.shape[1]):
						dK_dX[i,j,k] = (X[i][k] - X[j][k]) * kern.K(X[i,:][None],X[j,:][None])
			dK_dX = np.sum(dK_dX, axis=1)
			dlogl_dX = np.dot(dlogl_dK, dK_dX)		
			return -dlogl_dX.flatten(1)			

		# x, flog, function_eval, status = GPy.inference.optimization.SCG(f, grad, self.X.flatten(1))
		#latent_space = np.reshape(x,(2,N)).T
		result = minimize(f, X, method='BFGS', options={'disp': True}) #jac=grad
		latent_space = result.x.reshape(-1,Q)		
		return latent_space

