import numpy as np

class RBF(object):
#	Equivalent to:
#       cov = np.zeros((a.shape[0],b.shape[0]))
#       for i in range(0,a.shape[0]):
#           for j in range(0,b.shape[0]):
#               cov[i][j] = np.exp(-.5 * (1/lengthscale**2) * ((a[i]-b[j])**2))
#       return cov

    def __init__(self, lengthscale=1, sig_var=1, noise_var=1):
	self.lengthscale = lengthscale
	self.sig_var = sig_var
	self.noise_var = noise_var

    def compute(self, a, b):	       		
	sq_dist = np.sum(a**2, 1).reshape(-1, 1) + np.sum(b**2, 1) - 2*np.dot(a, b.T)
        cov = np.exp(-.5 * (1/(self.lengthscale**2)) * sq_dist)
	cov = (self.sig_var*cov)
	return cov

    def compute_noisy(self, a, b):	       		
	sq_dist = np.sum(a**2, 1).reshape(-1, 1) + np.sum(b**2, 1) - 2*np.dot(a, b.T)
        cov = np.exp(-.5 * (1/(self.lengthscale**2)) * sq_dist)
	noisy_cov = (self.sig_var*cov) + (self.noise_var*np.eye(cov.shape[1]))
	return noisy_cov

# Kernels to be completed

class Tanimoto(object):
	def __init__(self, lengthscale=None):
		self.lengthscale = lengthscale

	def compute(self, smilesA, smilesB):
		from rdkit import Chem

		molsA = [Chem.MolFromSmiles(compound) for compound in smilesA]
		fingerprintsA = [Chem.RDKFingerprint(compound, fpSize=2048) for compound in molsA]

		molsB = [Chem.MolFromSmiles(compound) for compound in smilesB]
		fingerprintsB = [Chem.RDKFingerprint(compound, fpSize=2048) for compound in molsB]

		from rdkit import DataStructs
		sims = []
		for i in xrange(len(smilesA)):
			sim_row = []
			for j in xrange(len(smilesB)):
				sim_row.append(DataStructs.FingerprintSimilarity(fingerprintsA[i],fingerprintsB[j], metric=DataStructs.TanimotoSimilarity))
			sims.append(sim_row)
		similarities = np.asarray(sims)
		distances = 1 - similarities
		cov = np.exp(-.5 * distances)
		return cov

	def compute_noisy(self, smilesA, smilesB):
		from rdkit import Chem

		molsA = [Chem.MolFromSmiles(compound) for compound in smilesA]
		fingerprintsA = [Chem.RDKFingerprint(compound, fpSize=2048) for compound in molsA]

		molsB = [Chem.MolFromSmiles(compound) for compound in smilesB]
		fingerprintsB = [Chem.RDKFingerprint(compound, fpSize=2048) for compound in molsB]

		from rdkit import DataStructs
		sims = []
		for i in xrange(len(smilesA)):
			sim_row = []
			for j in xrange(len(smilesB)):
				sim_row.append(DataStructs.FingerprintSimilarity(fingerprintsA[i],fingerprintsB[j], metric=DataStructs.TanimotoSimilarity))
			sims.append(sim_row)
		similarities = np.asarray(sims)
		distances = 1 - similarities
		cov = np.exp(-.5 * distances)
		cov = cov + np.eye(distances.shape[1])
		return cov


class ARD(object):
	def __init__(self, params):
		self.params = params

class Matern(object):
	def __init__(self, params):
		self.params = params

class Graph(object):
	def __init__(self, params):
		self.params = params
