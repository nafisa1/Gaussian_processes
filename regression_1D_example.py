import numpy as np
import kernels
import regression

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Our model does not know the true function values for the test points, but we will plot them to get an idea of how the model is performing

# Key for plots:
#	solid blue line - true output function
#	red crosses - training points
#	dotted red line - posterior mean (predicted output function)
#	shaded blue region - 95% uncertainty region

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# The prior is generated from the test values
X_test = np.vstack(np.linspace(0, 5, 100))
# We are generating the true output values over the range of test (and training) points for comparison purposes
true_f = np.sin(0.9*X_test)

# Generate training values
# The posterior is generated using these values 
X_train = np.vstack(5*np.random.rand(10, 1))
train_noise = np.vstack(0.05*np.random.randn(10, 1))
Y_train = np.sin(0.9*X_train) + train_noise

# Set up model
kern = kernels.RBF(lengthscale=0.2, sig_var=1, noise_var=0.1)
m = regression.Regression(X_test, X_train, Y_train, add_noise=0, kernel=kern, Ytest=true_f) # Optional: kernel, normalize

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Plot the prior and posterior (and random draws from both)

# Note how the posterior uncertainty is large for the regions of the input space where there are no training values. The posterior mean visibly deviates from the true mean in these regions.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

m.plot_prior()
m.plot_posterior()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
