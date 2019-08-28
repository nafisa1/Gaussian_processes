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

def test_function(name):
	print len(name)

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
    positive = 0
    negative = 0
    true_p = 0
    true_n = 0
    labels = []
    tpr = []
    fpr = [] # for roc plot
    
    for n in xrange(pred.shape[0]):
      if ytest[n] >= t:
        positive += 1
        labels.append(1)
        if pred[n] >= t:
          count += 1
          true_p +=1
      else: 
        negative += 1
        labels.append(0)
        if pred[n] <= t:
          count += 1
          true_n +=1

    correct = float(count)/pred.shape[0]
    sensitivity = float(true_p)/positive
    specificity = float(true_n)/negative
        
    print ('%d compounds out of %d (%f) were classified correctly. The sensitivity is %f (%d out of %d) and the specificity is %f (%d out of %d).' %(count,pred.shape[0],correct,sensitivity,true_p,positive,specificity,true_n,negative))
    
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, pred.flatten())
    import matplotlib.pyplot as plt
    
    plt.plot([0,1],[0,1], linestyle='--')
    plt.plot(fpr,tpr, marker='.')
    plt.show()
    if roc == True:
        return tpr,fpr,positive,true_p,negative,true_n#tp,tn,tp_correct,tn_correct

# Latin hypercube sampling

def LHS(parameters=2, n_choices=100, lower=[1.0,1.0], upper=[20.0,20.0], divisions=[12,12]): # 2 4 3 8
	
  import itertools
  scales = []
  for i in xrange(parameters):
    if i % 2 == 0:
      scale = np.linspace(lower[0],upper[0],divisions[0])
    else:
      scale = np.linspace(lower[1],upper[1],divisions[1])
    scales.append(scale)

  all_combs = np.asarray(list(itertools.product(*scales)))
#  return all_combs
  combinations = all_combs[np.random.randint(all_combs.shape[0], size=n_choices),:]
  return combinations

def enantiomers(smiles,output,names,descriptors=None):
    import kernels
    elim_ker = kernels.RBF()
    elim_cov = elim_ker.compute(smiles, smiles)
    new_descs = []
    new_smiles = []
    new_output = []
    new_names = []
    all_counts = []
    for i,row in enumerate(elim_cov):
        row_count = 0
        for j in xrange(i+1,len(row)):
            if row[j] == 1:
                row_count+=1
        all_counts.append(row_count)
           
    for x,number in enumerate(all_counts):
        if number == 0:
            if descriptors is not None:
                new_descs.append(descriptors[x])
            new_smiles.append(smiles[x])
            new_output.append(output[x])
            new_names.append(names[x])
    new_descs = np.asarray(new_descs)
    new_output = np.asarray(new_output)
    print len(smiles)-len(new_smiles),"enantiomers"
    print len(new_smiles),"remaining compounds"
    return new_smiles, new_output, new_names, new_descs
    
#		if self.pca == True:
#			Xtrain_num, W = GPy.util.linalg.pca(Xtrain_num, self.latent_dim)
#			jitter = 0.05*np.random.rand((Xtrain_num.shape[0]), (Xtrain_num.shape[1]))
#			jitter -= 0.025
#			Xtrain_num = Xtrain_num - jitter
#	
#			Xtest_num = np.dot(W,Xtest_num.T).T
#			jitter = 0.05*np.random.rand((Xtest_num.shape[0]), (Xtest_num.shape[1]))
#			jitter -= 0.025
#			Xtest_num = Xtest_num - jitter


