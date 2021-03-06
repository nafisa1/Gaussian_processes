import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

def jit_chol(cov, attempts=1000, print_jit=False):
    jitter = 0
    for i in xrange(attempts):
        try:
            cov_chol = np.linalg.cholesky(cov)
            break
        except:
            jitter = abs(np.mean(np.diag(cov)))*1e-2
            cov = cov + jitter*np.eye(cov.shape[0])
            if i == (attempts-1):
                print "Covariance matrix is not positive definite"
    if print_jit == True:
	print "jitter = ", jitter
    return cov_chol, jitter

def distance(a, b, sim_metric, circ_radius, circular):
  def get_fps(smiles, radius=2, circular=True):
    mols = [Chem.MolFromSmiles(compound) for compound in smiles]
    if circular is True:
      fingerprints = [AllChem.GetMorganFingerprint(compound, radius) for compound in mols]
    else:
      fingerprints = [Chem.RDKFingerprint(compound, fpSize=2048) for compound in mols]
	
    return fingerprints

  if isinstance(a[0], str) == True:	
		fingerprintsA = get_fps(a, circular=circular, radius=circ_radius)
		fingerprintsB = get_fps(b, circular=circular, radius=circ_radius) 

		sims = []

		for i in xrange(len(a)):
			sim_row = []
			for j in xrange(len(b)):			
				sim_row.append(sim_metric(fingerprintsA[i],fingerprintsB[j]))
			sims.append(sim_row)
		similarities = np.asarray(sims)
		distances = 1-similarities

  else:
    distances = np.absolute(np.sum(a, 1).reshape(-1, 1) - np.sum(b, 1))

  return distances

class Linear(object):
	def __init__(self, sim_metric=DataStructs.TanimotoSimilarity, lengthscale=1, sig_var=1, noise_var=1, datatype='string', circ_radius=2, circular=True):
		self.lengthscale = lengthscale
		self.sig_var = sig_var
		self.noise_var = noise_var
		self.datatype = datatype
		self.sim_metric = sim_metric
		self.circ_radius = circ_radius
		self.circular = circular

	def compute(self, a, b, noise=False):

		distances = distance(a, b, self.sim_metric, self.circ_radius, self.circular)		
		cov = self.lengthscale + self.sig_var*distances # lengthscale is bias in this case

		if noise==True:
			cov = cov + (self.noise_var*np.eye(cov.shape[0]))

		return cov


class RBF(object):
	def __init__(self, sim_metric=DataStructs.TanimotoSimilarity, lengthscale=1, sig_var=1, noise_var=1, datatype='string', circ_radius=2, circular=True):
		self.lengthscale = lengthscale
		self.sig_var = sig_var
		self.noise_var = noise_var
		self.datatype = datatype
		self.sim_metric = sim_metric
		self.circ_radius = circ_radius
		self.circular = circular

	def compute(self, a, b, noise=False):

		distances = distance(a, b, self.sim_metric, self.circ_radius, self.circular)
		sq_dist = distances**2
		cov = self.sig_var*np.exp(-.5 * sq_dist * (1/(self.lengthscale**2)))

		if noise==True:
			cov = cov + (self.noise_var*np.eye(cov.shape[0]))

		return cov

class Matern(object):
	def __init__(self, sim_metric=DataStructs.TanimotoSimilarity, nu=0, lengthscale=1, sig_var=1, noise_var=1, datatype='string', circ_radius=2, circular=True, data_type=str):

		self.nu = nu
		self.lengthscale = lengthscale
		self.sig_var = sig_var
		self.noise_var = noise_var
		self.datatype = datatype
		self.sim_metric = sim_metric
		self.circ_radius = circ_radius
		self.circular = circular
		self.data_type = data_type

	def compute(self, a, b, noise=False):
		
		distances = distance(a, b, self.sim_metric, self.circ_radius, self.circular)
		
		if self.nu==0:
			cov = self.sig_var*np.exp(-distances * (1/(self.lengthscale)))
		elif self.nu==1:
			cov = self.sig_var*((1+((3**0.5)*distances/self.lengthscale))*np.exp(-distances* (3**0.5) * (1/(self.lengthscale))))
		elif self.nu==2:
			cov = self.sig_var*(1+((5**0.5)*distances/self.lengthscale)+((5*(distances**2))/(3*(self.lengthscale**2))))*np.exp(-distances* (5**0.5) * (1/(self.lengthscale)))

		if noise==True:
			cov = cov + (self.noise_var*np.eye(cov.shape[0]))

		return cov

class RQ(object):
	def __init__(self, sim_metric=DataStructs.TanimotoSimilarity, lengthscale=0.5, noise_var=1, datatype='string', circ_radius=2, circular=True):
		self.lengthscale = lengthscale
		self.noise_var = noise_var
		self.datatype = datatype
		self.sim_metric = sim_metric
		self.circ_radius = circ_radius
		self.circular = circular

	def compute(self, a, b, noise=False):

		distances = distance(a, b, self.sim_metric, self.circ_radius, self.circular)
		sq_dist = distances ** 2
		cov = (1 + sq_dist)**(-self.lengthscale)

		if noise==True:
			cov = cov + (self.noise_var*np.eye(cov.shape[0]))

		return cov

class Periodic(object):
	def __init__(self, sim_metric=DataStructs.TanimotoSimilarity, lengthscale=1, sig_var=1, noise_var=1, datatype='string', circ_radius=2, circular=True):
		self.lengthscale = lengthscale
		self.sig_var = sig_var
		self.noise_var = noise_var
		self.datatype = datatype
		self.sim_metric = sim_metric
		self.circ_radius = circ_radius
		self.circular = circular

	def compute(self, a, b, noise=False):

		distances = distance(a, b, self.sim_metric, self.circ_radius, self.circular)
		cov = self.sig_var*np.exp(-2*((np.sin(distances/2))**2)/(self.lengthscale**2))

		if noise==True:
			cov = cov + (self.noise_var*np.eye(cov.shape[0]))

		return cov

class Composite(object):
	def __init__(self, kern1, kern2, noise_var=1, method='add', kern3=None, kern4=None, kern5=None, kern6=None):
		self.noise_var = noise_var
		self.kern1 = kern1
		self.kern2 = kern2
		self.kern3 = kern3
		self.kern4 = kern4
		self.kern5 = kern5
		self.kern6 = kern6
		self.method = 'add' #not changing properly

		self.kers = [self.kern1, self.kern2, self.kern3, self.kern4, self.kern5, self.kern6]

		n_kers = 0
		for item in self.kers:
			if item is not None:
				n_kers += 1
		self.n_kers = n_kers
	
	def compute(self, inputA, inputB, noise=False):
		covs = []
		if len(inputA) == 2:
			for x,item in enumerate(self.kers):
				if item is not None:
					item_cov = item.compute(inputA[x], inputB[x])
					covs.append(item_cov)
		else:
			for x,item in enumerate(self.kers):
				if item is not None:
					item_cov = item.compute(inputA, inputB)
					covs.append(item_cov)
		covs = np.asarray(covs)

		if self.method == 'add':
			cov = np.sum(covs, axis=0)
		elif self.method == 'multiply':
			cov = np.prod(covs, axis=0)

		if noise==True:
			cov = cov + (self.noise_var*np.eye(cov.shape[0]))

		return cov

class ARD(object):
	def __init__(self, params):
		self.params = params


class Graph(object):
	def __init__(self, params):
		self.params = params
