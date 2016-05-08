import numpy as np
import matplotlib.pyplot as plt

def plot_prior_1D(Xtest, Xtrain, Ytrain, test_cov, Ytest=None):

	# Manipulate X for plotting 
	X = np.hstack(Xtest)

	# Set prior mean function
	mean = np.zeros(Xtest.shape)

	s = np.sqrt(np.diag(test_cov))
	mean = np.reshape(mean, (-1,))
	print mean.shape
	print s.shape
		     
	# Plot true function, mean function and uncertainty   
	plt.figure()
	plt.xlim(min(X), max(X))
	plt.ylim(min(mean-(2*s)-(s/2)), max(mean+(2*s)+(s/2)))       

	if Ytest is not None:
		plt.plot(X, Ytest, 'b-', label='Y')
	plt.plot(X, mean, 'r--', lw=2, label='mean')
	plt.fill_between(X, mean-(2*s), mean+(2*s), color='#87cefa')
	plt.legend()
	
	# Plot draws from prior
	mean = mean.reshape(X.shape[0],1)
	f = mean + np.dot(test_cov, np.random.normal(size=(X.shape[0],10)))
	plt.figure()
	plt.xlim(min(X), max(X))
	plt.plot(X, f)
	plt.title('Ten samples')
	plt.show() 

def plot_posterior_1D(Xtest, Xtrain, Ytrain, p_mean, p_sd, cov_post, Ytest=None): 
		
	# Manipulate data for plotting
	mean_f = p_mean.flat
	p_sd = np.reshape(p_sd, (-1,))
	Xtest = np.hstack(Xtest)
		
	# Plot true function, predicted mean and uncertainty (2s), and training points
	plt.figure()
	plt.plot(Xtrain, Ytrain, 'r+', ms=20) # training points
	plt.xlim(min(Xtest), max(Xtest))
	plt.ylim(min(mean_f-(2*p_sd)-(p_sd/2)), max(mean_f+(2*p_sd)+(p_sd/2))) 
 
	if Ytest is not None:      
		plt.plot(Xtest, Ytest, 'b-', label='Y') # true function
	plt.plot(Xtest, mean_f, 'r--', lw=2, label='mean') # mean function
	plt.fill_between(Xtest, mean_f-(2*p_sd), mean_f+(2*p_sd), color='#87cefa') # uncertainty
	plt.legend()
	plt.show()
		
	# Plot 10 draws from posterior
	f = p_mean + np.dot(cov_post, np.random.normal(size=(Xtest.shape[0],10)))
	plt.figure()
	plt.xlim(min(Xtest), max(Xtest))
	plt.plot(Xtest, f)
	plt.plot(Xtrain, Ytrain, 'r+', ms=20) # new points
	plt.title('Ten samples')
	plt.show()
        
