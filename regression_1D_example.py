import numpy as np
import kernels
import regression

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Run this file to see an example of a Gaussian process model being trained to learn about a
# function of interest. 

# Initially, with no information we assume the function is zero everywhere. This is our 'prior'.
# The model learns using 'training points' (input values for which the function value is known).
# After training, we obtain the 'posterior'. The model can make predictions of the function 
# value at 'test points', input values for which the function value is unknown.

# In this example, we have defined the output function, so we do know what it looks like over 
# the input space. However, the model is only given the training points - the output function 
# is only plotted to get an idea of how the model is performing. Usually the output function 
# is not known.

# Key for plots:
#	solid blue line - true output function
#	red crosses - training points
#	dotted red line - posterior mean (predicted output function)
#	shaded blue region - 95% uncertainty region

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Generate test values
X_test = np.vstack(np.linspace(0, 5, 100))
# Output function (plotted for comparison only)
true_f = np.sin(0.9*X_test)

# Generate training inputs with noisy output values
# The posterior is generated using these values 
X_train = np.vstack(5*np.random.rand(10, 1))
train_noise = np.vstack(0.05*np.random.randn(10, 1))
Y_train = np.sin(0.9*X_train) + train_noise

# Set up model
kern = kernels.RBF(lengthscale=0.2, sig_var=1, noise_var=0.1)
m = regression.Regression(X_test, X_train, Y_train, add_noise=0.1, kernel=kern, Ytest=true_f) # Optional: kernel, normalize

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Plot the prior and posterior (and random draws from both)

# Note how the posterior uncertainty is large for the regions of the input space where there
# are no training values. The posterior mean visibly deviates from the true mean in these 
# regions. Close to the training points, the posterior mean is a good representation of the
# true output function.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

m.plot_prior()
m.plot_posterior()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
