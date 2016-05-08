import numpy as np
import kernels
import regression

# Produce input range and true function values
#
# Our model does not know the true function values for the test points, but we will plot them to get an idea of how the model is performing

# The prior is generated from the test values
X_test = np.vstack(np.linspace(0, 6, 50))
# We are generating the true function values over the range of test (and training) points for comparison purposes
true_f = np.sin(0.9*X_test)

# Generate training values
# The posterior is generated using these values 
X_train = np.vstack(5*np.random.rand(40, 1))
train_noise = np.vstack(0.05*np.random.randn(40, 1))
Y_train = np.sin(0.9*X_train) + train_noise

kern = kernels.RBF(lengthscale=0.5, noise_var=0.2)
m = regression.Regression(X_test, X_train, Y_train, add_noise=0, kernel=kern, Ytest=true_f) # Optional: kernel, normalize

m.plot_prior()
m.plot_posterior()
