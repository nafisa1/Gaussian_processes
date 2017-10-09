import numpy as np
import utils
import xlrd
import model

class Experiment(object):
	
	def get_data(self,filename, id_name, smiles_name, output_name, descriptor_names, output_type='pic50', upper_threshold=None):
	    workbook = xlrd.open_workbook(filename, on_demand=True)
	    sheet = workbook.sheet_by_index(0)
	    
	    properties = sheet.row_values(0)
	    reordered_descriptor_names = []
	    descriptors = []
	    for i,item in enumerate(properties):
		if item == id_name:
		    names = sheet.col_values(i)[1:]
		elif item == output_name:
		    output = sheet.col_values(i)[1:]
		elif item == smiles_name:
		    smiles = sheet.col_values(i)[1:]
		elif str(item) in descriptor_names:
		    print item, i
		    reordered_descriptor_names.append(item)
		    descriptors.append(sheet.col_values(i)[1:])
	    descriptors = np.asarray(descriptors).T
	    print len(names),"compounds initially"

	    new_names = []
	    new_output = []
	    new_smiles = []
	    new_desc = []

	    for i, name in enumerate(names):
		if name != '' and output[i] != '' and output[i] < upper_threshold:
		    new_names.append(name[4:])
		    new_output.append(output[i])
		    new_desc.append(list(descriptors[i]))
		    new_smiles.append(str(smiles[i]).split()[0])
	    print len(new_names),"compounds left after removing missing compounds and inactive compounds"
	    
	    zipped = zip(new_names, new_output, new_smiles, new_desc)
	    zipped.sort()
	    new_names, new_output, new_smiles, new_desc = zip(*zipped)
	    new_desc = np.asarray(new_desc)
	    print 'Compounds have been sorted by EVOTEC ID'

	    if output_type == 'pic50':
		pic50s = utils.pIC50(new_output, -9)

	    self.smiles, self.output, self.descriptors = utils.enantiomers(new_smiles, pic50s, new_desc)
	    return self.smiles, self.output, self.descriptors

	def bayes_opt(self,training_size, test_size, ker, acquisition_function, noise=0.01, number_runs=None):
	    start_test = len(self.output)-test_size
	    modopt = model.Model(n_kers=2, Xtrain=[self.descriptors[:training_size],self.smiles[:training_size]],Xtest=[self.descriptors[training_size:start_test],self.smiles[training_size:start_test]], Ytrain=self.output[:training_size], Ytest=self.output[training_size:start_test], kernel=ker) 
	    r_sq = []
	    modtest = model.Model(n_kers=2, Xtrain=modopt.Xtrain, Xtest=[self.descriptors[start_test:],self.smiles[start_test:]], Ytrain=modopt.Ytrain, Ytest=self.output[start_test:], kernel=ker) 
	    modtest.kernel.noise_var = noise
	    modtest.hyperparameters(print_vals=False)
	    regtest = modtest.regression()
	    regtest.plot_by_index()
	    r_sq.append(regtest.r_squared())
	    print "r squared",r_sq 
	    
	    for i in xrange(number_runs):
		if i%10 == 0:
		    print "Iteration",i,"..."
		modopt.acq_func = acquisition_function
		newx = modopt.optimization()  
		modtest.Xtrain = modopt.Xtrain
		modtest.Ytrain = modopt.Ytrain 
		modtest.hyperparameters(print_vals=False)
		regtest = modtest.regression()
		print "r squared",regtest.r_squared()
		r_sq.append(regtest.r_squared())
		print len(modopt.Ytrain)
	    return modopt,modtest,r_sq
