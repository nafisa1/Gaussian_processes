import numpy as np
import kernels
import regression_1D

# Produce input range and true function values
# Our model does not know the true function values, but we will plot them to get an idea of how the model is performing
# The prior is generated from the input range
input_range = np.vstack(np.linspace(0, 5, 50))
true_f = np.sin(0.9*input_range)

# Calculate noise of function values and noise variance
# Noise variance is a parameter of the regression model
noise = np.vstack(0.4*np.random.randn(50, 1))
noise_variance = np.var(noise)

# Generate test values (data)
# The posterior is generated using the test 
X_train = np.vstack(5*np.random.rand(5, 1))
train_noise = np.vstack(0.05*np.random.randn(5, 1))
Y_train = np.sin(0.9*X_train) + train_noise

kern = kernels.RBF()
m = regression_1D.Regression(input_range, true_f, X_train, Y_train, kernel=kern) # Optional: kernel, normalize
m.plot_prior()
m.plot_posterior(X_train, Y_train)
