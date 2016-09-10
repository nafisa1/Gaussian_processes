import numpy as np
import matplotlib.pyplot as plt

def plot_prior_1D(Xtest, test_cov, Ytest=None):

	# Manipulate X for plotting 
	X = np.hstack(Xtest)

	# Set prior mean function
	mean = np.zeros(Xtest.shape)

	s = np.sqrt(np.diag(test_cov))
	mean = np.reshape(mean, (-1,))
		     
	# Plot true function, mean function and uncertainty   
	ax1 = plt.subplot(211)
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
	ax2 = plt.subplot(212, sharex=ax1)
	plt.plot(X, f)
	plt.title('Ten samples')
	plt.tight_layout()
	plt.show() 

def plot_posterior_1D(Xtest, Xtrain, Ytrain, p_mean, p_sd, cov_post, Ytest=None): 
		
	# Manipulate data for plotting
	mean_f = p_mean.flat
	p_sd = np.reshape(p_sd, (-1,))
	Xtest = np.hstack(Xtest)
		
	# Plot true function, predicted mean and uncertainty (2s), and training points
	ax1 = plt.subplot(211)
	plt.plot(Xtrain, Ytrain, 'r+', ms=20) # training points
	plt.xlim(min(Xtest), max(Xtest))
	plt.ylim(min(mean_f-(2*p_sd)-(p_sd/2)), max(mean_f+(2*p_sd)+(p_sd/2))) 
 	if Ytest is not None:      
		plt.plot(Xtest, Ytest, 'b', label='Y') # true function
	plt.plot(Xtest, mean_f, 'r--', lw=2, label='mean') # mean function
	plt.fill_between(Xtest, mean_f-(2*p_sd), mean_f+(2*p_sd), color='#87cefa') # uncertainty
	plt.legend()
		
	# Plot 10 draws from posterior
	f = p_mean + np.dot(cov_post, np.random.normal(size=(Xtest.shape[0],10)))
	ax2 = plt.subplot(212, sharex=ax1)
	plt.xlim(min(Xtest), max(Xtest))
	plt.plot(Xtest, f)
	plt.plot(Xtrain, Ytrain, 'r+', ms=20) # new points
	plt.title('Ten samples')
	plt.tight_layout()
	plt.show()
        
def plot_prior_2D(Xtest, test_cov, Ytest=None):
	from mpl_toolkits.mplot3d import Axes3D

	prior_s = np.sqrt(np.diag(test_cov))
		
	# Create prior mean vector and vectors bounding the 95% uncertainty region
	n = Xtest[:,0].shape[0]
	prior_mean = np.zeros(n)
	upper = prior_mean + (2*prior_s)        
	lower = prior_mean - (2*prior_s) 
	       
	# Plot mean points and uncertainty
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1, projection = '3d')
	if Ytest is not None:      
		ax.scatter(Xtest[:,0], Xtest[:,1], Ytest, c= 'g', marker='^')#, label='Y') # true function
	ax.scatter(Xtest[:,0], Xtest[:,1], prior_mean) 
	ax.scatter(Xtest[:,0], Xtest[:,1], upper, c= 'r')
	ax.scatter(Xtest[:,0], Xtest[:,1], lower, c= 'r')
	plt.show() 

def plot_posterior_2D(Xtest, Xtrain, Ytrain, p_mean, p_sd, acq1, Ytest=None):
	from mpl_toolkits.mplot3d import Axes3D

	upper = p_mean + (2*p_sd)
	lower = p_mean - (2*p_sd)
		
	# Plot posterior mean points and uncertainty
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1, projection = '3d')
	if Ytest is not None: 
		ax.scatter(Xtest[:,0], Xtest[:,1], Ytest, c= 'g', marker='^')#, label='Y') # true function
	ax.scatter(Xtest[:,0], Xtest[:,1], p_mean) 
	ax.scatter(Xtest[:,0], Xtest[:,1], upper, c='r')
	ax.scatter(Xtest[:,0], Xtest[:,1], lower, c='r')
	ax.scatter(Xtrain[:,0], Xtrain[:,1], Ytrain, c='g',marker='^', s = 70)
	plt.show()

def plot_acq(Xtest, acq, p_mean, p_sd, Ytest=None):
# Plot posterior of test set with acquisition function underneath

	# Plot true function, predicted mean and uncertainty (2s), and training points
	ax1 = plt.subplot(211)
	plt.title("Posterior over Test Set")

	# Manipulate data for plotting
	mean_f = p_mean.flat
	p_sd = np.reshape(p_sd, (-1,))

#	plt.xlim(min(Xtest), max(Xtest))		
#	plt.plot(Xtrain, Ytrain, 'b+', ms=10) # training points

 	if Ytest is not None:      
		plt.plot(Xtest, Ytest, 'ro', label='Y') # true function

	Xtest,mean_f = zip(*sorted(zip(Xtest,mean_f),key=lambda Xtest: Xtest[0]))
	X_next = Xtest[np.argmax(acq)]
	plt.ylabel('Predicted log(IC50)')
	plt.plot(X_next, mean_f[np.argmax(acq)], 'o', color='#ff3333')
	plt.plot(Xtest, mean_f, '--', color='#000099', label='Posterior mean') # mean function
	plt.fill_between(Xtest, mean_f-(2*p_sd), mean_f+(2*p_sd), color='#e6e6ff', label="95% credibility interval") # uncertainty
	plt.legend(loc=0, fontsize='medium')
	
	ax2 = plt.subplot(212, sharex=ax1)
	plt.title("Acquisition function")
	Xtest,acq = zip(*sorted(zip(Xtest,acq),key=lambda Xtest: Xtest[0]))
	plt.xlabel('First principal component')
	plt.ylabel('Acquisition function')
	plt.plot(Xtest, acq, color='#800000')
	plt.plot(X_next, max(acq), 'o', color='#ff3333')
	plt.ylim(0, max(acq)+0.2)

	plt.tight_layout()
	plt.show()
