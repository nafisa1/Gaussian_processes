import numpy as np
from rdkit import Chem
from rdkit import DataStructs
import utils 

class RBF(object):
#	Equivalent to:
#       cov = np.zeros((a.shape[0],b.shape[0]))
#       for i in range(0,a.shape[0]):
#           for j in range(0,b.shape[0]):
#               cov[i][j] = np.exp(-.5 * (1/lengthscale**2) * ((a[i]-b[j])**2))
#       return cov

	def __init__(self, lengthscale=1, sig_var=1, noise_var=1, datatype='numerical'):
		self.lengthscale = lengthscale
		self.sig_var = sig_var
		self.noise_var = noise_var
		self.datatype = datatype

	def compute(self, a, b):	       		
		sq_dist = np.sum(a**2, 1).reshape(-1, 1) + np.sum(b**2, 1) - 2*np.dot(a, b.T)
	        cov = np.exp(-.5 * (1/(self.lengthscale**2)) * sq_dist)
		cov = (self.sig_var*cov)
		return cov

class OU_num(object):

	def __init__(self, lengthscale=1, sig_var=1, noise_var=1, datatype='numerical'):
		self.lengthscale = lengthscale
		self.sig_var = sig_var
		self.noise_var = noise_var
		self.datatype = datatype

	def compute(self, a, b):	       		
		distances = np.absolute(np.sum(a, 1).reshape(-1, 1) - np.sum(b, 1))
		cov = self.sig_var*np.exp(-distances * (1/(self.lengthscale)))
		return cov

class SMILES_RBF(object):
	def __init__(self, metric=DataStructs.TanimotoSimilarity, lengthscale=1, sig_var=1, noise_var=1, datatype='string'):
		self.lengthscale = lengthscale
		self.sig_var = sig_var
		self.noise_var = noise_var
		self.datatype = datatype
		self.metric = metric

	def compute(self, smilesA, smilesB):

		fingerprintsA = utils.get_fps(smilesA)
		fingerprintsB = utils.get_fps(smilesB) 

		sims = []
		for i in xrange(len(smilesA)):
			sim_row = []
			for j in xrange(len(smilesB)):
				sim_row.append(DataStructs.FingerprintSimilarity(fingerprintsA[i],fingerprintsB[j], metric=self.metric))
			sims.append(sim_row)
		similarities = np.asarray(sims)
		distances = 1 - similarities
		sq_dist = distances**2
		cov = self.sig_var*np.exp(-.5 * sq_dist * (1/(self.lengthscale**2)))
		return cov

class Matern(object):
	def __init__(self, metric=DataStructs.TanimotoSimilarity, nu=0, lengthscale=1, sig_var=1, noise_var=1, datatype='string'):
		self.metric = metric 
		self.nu = nu
		self.lengthscale = lengthscale
		self.sig_var = sig_var
		self.noise_var = noise_var
		self.datatype = datatype
		self.metric = metric

	def compute(self, smilesA, smilesB):

		fingerprintsA = utils.get_fps(smilesA)
		fingerprintsB = utils.get_fps(smilesB) 

		sims = []
		for i in xrange(len(smilesA)):
			sim_row = []
			for j in xrange(len(smilesB)):
				sim_row.append(DataStructs.FingerprintSimilarity(fingerprintsA[i],fingerprintsB[j], metric=self.metric))
			sims.append(sim_row)
		similarities = np.asarray(sims)
		distances = 1 - similarities
		
		if self.nu==0:
			cov = self.sig_var*np.exp(-distances * (1/(self.lengthscale)))
		elif self.nu==1:
			cov = self.sig_var*((1+((3**0.5)*distances/self.lengthscale))*np.exp(-distances* (3**0.5) * (1/(self.lengthscale))))
		elif self.nu==2:
			cov = self.sig_var*(1+((5**0.5)*distances/self.lengthscale)+((5*(distances**2))/(3*(self.lengthscale**2))))*np.exp(-distances* (5**0.5) * (1/(self.lengthscale)))
		return cov

class RQ(object):
	def __init__(self, metric=DataStructs.TanimotoSimilarity, lengthscale=0.5, noise_var=1, datatype='string'):
		self.lengthscale = lengthscale
		self.noise_var = noise_var
		self.datatype = datatype
		self.metric = metric

	def compute(self, smilesA, smilesB):

		molsA = [Chem.MolFromSmiles(compound) for compound in smilesA]
		fingerprintsA = [Chem.RDKFingerprint(compound, fpSize=2048) for compound in molsA]

		molsB = [Chem.MolFromSmiles(compound) for compound in smilesB]
		fingerprintsB = [Chem.RDKFingerprint(compound, fpSize=2048) for compound in molsB]

		sims = []
		for i in xrange(len(smilesA)):
			sim_row = []
			for j in xrange(len(smilesB)):
				sim_row.append(DataStructs.FingerprintSimilarity(fingerprintsA[i],fingerprintsB[j], metric=self.metric))
			sims.append(sim_row)
		similarities = np.asarray(sims)
		distances = 1 - similarities
		sq_dist = distances ** 2
		cov = (1 + sq_dist)**(-self.lengthscale)
		return cov

class Composite(object):
	def __init__(self, kern1, kern2, noise_var=1, kern3=None, kern4=None, kern5=None, kern6=None):
		self.noise_var = noise_var
		self.kern1 = kern1
		self.kern2 = kern2
		self.kern3 = kern3
		self.kern4 = kern4
		self.kern5 = kern5
		self.kern6 = kern6

		self.kers = [self.kern1, self.kern2, self.kern3, self.kern4, self.kern5, self.kern6]

		nkers = 0
		for item in self.kers:
			if item is not None:
				nkers += 1
		self.nkers = nkers
	
	def compute(self, numA=None, numB=None, smilesA=None, smilesB=None):
		covs = []
		for i in xrange(self.nkers):
			for item in self.kers:
					#if item is not None:
				if item.datatype == 'numerical':
					item_cov = item.compute(numA, numB)
					covs.append(item_cov)
				else:
					item_cov = item.compute(smilesA, smilesB)
					covs.append(item_cov)
		covs = np.asarray(covs)
		cov = np.sum(covs, axis=0)
		return cov

class ARD(object):
	def __init__(self, params):
		self.params = params


class Graph(object):
	def __init__(self, params):
		self.params = params
