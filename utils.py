import numpy as np
from rdkit import Chem

def remove_zero_cols(A, B=None):
        # Calculate number of observations, N
	N = A.shape[0]

	# Remove columns of zeros
	zeros = np.zeros((1,N))[0]
	A_transpose = A.T
	non_zero_cols = []

	if B==None:
		for row in A_transpose:
			if np.array_equal(row, zeros) == False:
 	      			non_zero_cols.append(row)
       
		non_zero_cols = np.array(non_zero_cols)
		A = non_zero_cols.T

		return A

	else:
		testT = B.T
		selected_cols_test = []
		
		for row, rowtest in zip(A_transpose,testT):
			if np.array_equal(row, zeros) == False and np.array_equal(rowtest, zeros) == False:
				non_zero_cols.append(row)
		        	selected_cols_test.append(rowtest)

		non_zero_cols = np.array(non_zero_cols)
		A = non_zero_cols.T

		selected_cols_test = np.array(selected_cols_test)
		B = selected_cols_test.T
			
		return A, B

def remove_identical(A, B=None):
	# To examine each column, take the transpose
	transpose = A.T
	n = transpose.shape[1]
	non_identical = []

	# Take each row of A.T and create array equal in length, where all elements are set to the first element of the original row
	# Compare to identify rows where all elements are the same

	if B==None:
		for row in transpose:
	        	check = [] + n*[row[0]]
	        	check = np.asarray(check)

    	        	if np.array_equal(row,check) == False:
				non_identical.append(np.ndarray.tolist(row))

		return np.asarray(non_identical).T
	
	else:
		non_identB = []
		Btranspose = B.T

		for row, rowB in zip(transpose,Btranspose):
			check = [] + n*[row[0]]
	        	check = np.asarray(check)
			check_test = [] + n*[rowB[0]]
	        	check_test = np.asarray(check_test)

    	        	if np.array_equal(row,check) == False and np.array_equal(rowB,check_test) == False:
				non_identical.append(np.ndarray.tolist(row))
				non_identB.append(np.ndarray.tolist(rowB))
            
		return np.asarray(non_identical).T, np.asarray(non_identB).T

def normalize_centre(A, B=None):
	# Get mean and standard deviation
	A_mu = np.vstack(np.mean(A, axis=0))
	A_sd = np.vstack(A.std(axis=0))
#	print A_mu
	np.set_printoptions(threshold=np.nan)
#	print A_sd.shape
#	print A.shape

	# Centre and normalize array
	if B is None:
		A_centred = A.T - A_mu
		Acent_normalized = A_centred/A_sd
		final_array = Acent_normalized.T   

	# Centre and normalize second array using mean and standard deviation of first array
	else:
		B_centred = B.T - A_mu
		Bcent_normalized = B_centred/A_sd
		final_array = Bcent_normalized.T

	return final_array		

def centre(A, B=None):
	# Get mean
	if A.shape[1] > 1:
		A_mu = np.vstack(np.mean(A, axis=0))

	else:
		A_mu = np.mean(A, axis=0)

	# Centre array
	if B is None:
		A_centred = A.T - A_mu                    
		centred_array = A_centred.T   

	# Centre second array using mean of first array
	else:
		B_centred = B.T - A_mu
		centred_array = B_centred.T

	return centred_array	

def get_SMILES(filename, n_lines):
	with open(filename,'r') as f:
		names = []
		filecontents = f.readlines()
		count = 0
		for line in filecontents:
			lin = line.strip('\n')      
			items = lin.split()
			names.append(str(items[1]))
			count +=1
			if count == n_lines:
				break
	return names

def remove_500s(X, Y, smiles):
	newX = []
	newY = []
	newSmiles = []

	for Xrow, Ynum, smile in zip(X, Y, smiles):
		if Ynum != 50000:
			newX.append(Xrow)
			newY.append(Ynum)
			newSmiles.append(smile)
            
	return np.asarray(newX), np.asarray(newY), newSmiles


def shuffle(X, Y, split=0.8, smiles=None, prior=None):
	p = np.random.permutation(X.shape[0])
	X = X[p]
	Y = Y[p]
	
	if smiles is not None:
		smiles = [smiles[i] for i in p]
		smiles_train = smiles[:int(round(split*X.shape[0]))]
		smiles_test = smiles[int(round(split*X.shape[0])):]
	if prior is not None:
		prior = prior[p]
		prior_train = prior[:(split*X.shape[0]),:]
		prior_test = prior[(split*X.shape[0]):,:]
					
	
	Xtrain = X[:int(round(split*X.shape[0])),:]
	Ytrain = Y[:int(round(split*X.shape[0]))]
	Xtest = X[int(round(split*X.shape[0])):,:]
	Ytest = Y[int(round(split*X.shape[0])):]
	
	if smiles is not None and prior is not None:
		return Xtrain, Xtest, Ytrain, Ytest, smiles_train, smiles_test, prior_train, prior_test

	elif smiles is not None and prior is None:
		return Xtrain, Xtest, Ytrain, Ytest, smiles_train, smiles_test

	elif smiles is None and prior is not None:
		return Xtrain, Xtest, Ytrain, Ytest, prior_train, prior_test

	else:
		return Xtrain, Xtest, Ytrain, Ytest

def get_fps(smiles):
	mols = [Chem.MolFromSmiles(compound) for compound in smiles]
	fingerprints = [Chem.RDKFingerprint(compound, fpSize=2048) for compound in mols]
	
	return fingerprints

# Latin hypercube sampling

class LHS(object):
	def __init__(self, kernel, parameters=3, n_choices=15, lower=0.5, upper=2.5, divisions=6):
		self.kernel = kernel
		self.parameters = parameters
		self.divisions = divisions
		self.lower = lower
		self.upper = upper

		a = np.linspace(lower,upper,divisions)
		options = a.shape[0]

		full = np.zeros(((options**3),parameters))
		full[:,0] = np.concatenate(((options**2)*[a[0]], (options**2)*[a[1]], (options**2)*[a[2]], (options**2)*[a[3]], (options**2)*[a[4]], (options**2)*[a[5]]))
		full[:,1] = np.concatenate(options*[np.concatenate((options*[a[0]], options*[a[1]], options*[a[2]], options*[a[3]], options*[a[4]], options*[a[5]]))])
		full[:,2] = np.concatenate((options**2)*[a])

		self.combinations = full[np.random.randint(full.shape[0], size=n_choices),:]

	def compute(self, Xtest, Xtrain, Ytrain, Ytest):
		r_sq = []
		kern = self.kernel
		import regression
		regr = regression.Regression(Xtest, Xtrain, Ytrain, add_noise=0, kernel=kern, Ytest=Ytest)
		init_rsq = regr.r_squared()

		for i in xrange(self.divisions):
			kern.lengthscale, kern.sig_var, kern.noise_var = self.combinations[i][0], self.combinations[i][1], self.combinations[i][2]
			regr = regression.Regression(Xtest, Xtrain, Ytrain, kernel=kern, Ytest=Ytest)
			r_sq.append(regr.r_squared())
		
		if max(r_sq) > init_rsq:
			ind = np.argmax(r_sq)
			best = self.combinations[ind]

			print "The new kernel hyperparameters are: lengthscale=",best[0],", power=",best[1]," and noise variance=",best[2],"."
		
		else:
			best = np.array((self.kernel.lengthscale, self.kernel.sig_var, self.kernel.noise_var))
			print "The kernel hyperparameters will remain unchanged."

		return best[0], best[1], best[2]
