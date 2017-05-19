import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

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
	np.set_printoptions(threshold=np.nan)

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

def get_SMILES(filename):
	with open(filename,'r') as f:
		names = []
		filecontents = f.readlines()
		for line in filecontents:
			lin = line.strip('\n') # may need i = lin.split(),append(str(i[1]))     
			names.append(str(lin))
	return names

def get_sdf_property(filename, sdf_property):
	molecules = [x for x in Chem.SDMolSupplier(filename)]
	sdf_property_values = []

	for x in molecules:
		atoms = x.GetAtoms()
		props = x.GetPropNames()
		sdf_property_values.append(x.GetProp(sdf_property))

	return sdf_property_values

def remove_number(X, Y, smiles, number):
	newX = []
	newY = []
	newSmiles = []

	for Xrow, Ynum, smile in zip(X, Y, smiles):
		if Ynum != number:
			newX.append(Xrow)
			newY.append(Ynum)
			newSmiles.append(smile)
            
	return np.asarray(newX), np.asarray(newY), newSmiles


def shuffle(Y, split=0.8, X=None, smiles=None, prior=None):
	p = np.random.permutation(Y.shape[0])
	Y = Y[p]
	
	if X is not None:
		X = X[p]
		Xtrain = X[:int(round(split*X.shape[0])),:]	
		Xtest = X[int(round(split*X.shape[0])):,:]

	if smiles is not None:
		smiles = [smiles[i] for i in p]
		smiles_train = smiles[:int(round(split*Y.shape[0]))]
		smiles_test = smiles[int(round(split*Y.shape[0])):]
	if prior is not None:
		prior = prior[p]
		prior_train = prior[:(split*Y.shape[0]),:]
		prior_test = prior[(split*Y.shape[0]):,:]
					
	Ytrain = np.asarray(Y[:int(round(split*Y.shape[0]))])
	Ytest = np.asarray(Y[int(round(split*Y.shape[0])):])
	
	if smiles is not None and prior is not None:
		if X is not None:
			return Xtrain, Xtest, Ytrain, Ytest, smiles_train, smiles_test, prior_train, prior_test
		else:
			return Ytrain, Ytest, smiles_train, smiles_test, prior_train, prior_test


	elif smiles is not None and prior is None:
		if X is not None:
			return Xtrain, Xtest, Ytrain, Ytest, smiles_train, smiles_test
		else:
			return Ytrain, Ytest, smiles_train, smiles_test

	elif smiles is None and prior is not None:
		if X is not None:
			return Xtrain, Xtest, Ytrain, Ytest, prior_train, prior_test
		else:
			return Ytrain, Ytest, prior_train, prior_test

	else:
		if X is not None:
			return Xtrain, Xtest, Ytrain, Ytest
		else:
			return Ytrain, Ytest

def get_fps(smiles, radius=2, circular=True):
	mols = [Chem.MolFromSmiles(compound) for compound in smiles]

	if circular is True:
		fingerprints = [AllChem.GetMorganFingerprint(compound, radius) for compound in mols]
	else:
		fingerprints = [Chem.RDKFingerprint(compound, fpSize=2048) for compound in mols]
	
	return fingerprints

def pIC50(values, power):
	values = np.asarray(values)
	pIC50s = -np.log10(values*(10**power))
	return pIC50s

def classif(pred,ytest,t,roc=False):
    count = 0
    tp = 0
    tn = 0
    tp_correct = 0
    tn_correct = 0
    for n in xrange(pred.shape[0]):
        if ytest[n] >= t:
            tp += 1
            if pred[n] >= t:
                count += 1
                tp_correct +=1
        else: 
            tn += 1
            if pred[n] <= t:
                count += 1
                tn_correct +=1

    correct = float(count)/pred.shape[0]
    sensitivity = float(tp_correct)/tp
    specificity = float(tn_correct)/tn
        
    print ('%d compounds out of %d (%f) were classified correctly. The sensitivity is %f (%d out of %d) and the specificity is %f (%d out of %d).' %(count,pred.shape[0],correct,sensitivity,tp_correct,tp,specificity,tn_correct,tn))
    if roc == True:
        return tp,tn

# Latin hypercube sampling

def LHS(parameters=2, n_choices=5, lower=[0.01,1], upper=[10.0,7.0], divisions=[11,11]):
	
	import itertools
	scales = []
	for i in xrange(parameters):
		if i % 2 == 0:
			scale = np.linspace(lower[0],upper[0],divisions[0])
		else:
			scale = np.linspace(lower[1],upper[1],divisions[1])
		scales.append(scale)

	all_combs = np.asarray(list(itertools.product(*scales)))

	combinations = all_combs[np.random.randint(all_combs.shape[0], size=n_choices),:]
	return combinations
