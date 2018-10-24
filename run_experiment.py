import numpy as np
import utils
import xlrd
import model
import cross_validation

class Experiment(object):
	
	# EXTRACT COMPOUNDS, DESCRIPTORS AND ACTIVITIES FROM .CSV FILE
	def get_data(self,filename, id_name, smiles_name, output_name, descriptor_names, output_type='pic50', upper_threshold=None):
	    workbook = xlrd.open_workbook(filename, on_demand=True)
	    sheet = workbook.sheet_by_index(0)
	    
	    # FIELD NAMES ARE FIRST ROW
	    properties = sheet.row_values(0)
        
        # DESCRIPTORS ARE REORDERED ACCORDING TO THE ORDER THEY APPEAR IN THE FILE
	    reordered_descriptor_names = []
	    descriptors = []
	    
	    # IF A FIELD NAME MATCHES A REQUIRED PROPERTY, CREATE A VARIABLE FOR THAT COLUMN (VALUES ONLY)
	    for i,item in enumerate(properties):
		if item == id_name:
		    names = sheet.col_values(i)[1:]
		elif item == output_name:
		    output = sheet.col_values(i)[1:]
		elif item == smiles_name:
		    smiles = sheet.col_values(i)[1:]
		    
		# PRINT DESCRIPTOR NAME AND ROW NUMBER, CREATE NEW LIST AND ARRAY OF NAMES AND VALUES
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

	    # ONLY KEEP COMPOUNDS WITH NAME, OUTPUT, ALL DESCRIPTOR VALUES AND OUTPUT BELOW THRESHOLD
	    for i, name in enumerate(names):
        	if name != '' and output[i] != '' and '' not in descriptors[i] and output[i] < upper_threshold:
		        new_names.append(name[4:])
		        new_output.append(output[i])
		        new_desc.append(list(descriptors[i]))
		        new_smiles.append(str(smiles[i]).split()[0])

	    print len(new_names),"compounds left after removing missing compounds and inactive compounds"

	    # SORT REMAINING NAMES, OUTPUT AND INPUT ACCORDING TO EVOTEC ID	    
	    zipped = zip(new_names, new_output, new_smiles, new_desc)
	    zipped.sort()
	    new_names, new_output, new_smiles, new_desc = zip(*zipped)
	    new_desc = np.asarray(new_desc)

	    print 'Compounds have been sorted by EVOTEC ID'

	    # CONVERT IC50 TO PIC50
	    if output_type == 'pic50':
		    pic50s = utils.pIC50(new_output, -6)

            self.smiles, self.output, self.names, self.descriptors = new_smiles, pic50s, new_names, new_desc
	    # REMOVE ENANTIOMEERS 
	    # self.smiles, self.output, self.names, self.descriptors = utils.enantiomers(new_smiles, pic50s, new_names, new_desc)

	    return self.smiles, self.output, self.names, self.descriptors

	def bayes_opt(self,training_size, test_size, ker, acquisition_function, noise=0.01, number_runs=None, end_train=None, print_interim=False):

	    # POINT WHERE TEST SET STARTS IS DEFINED AS THE LAST x COMPOUNDS WHERE x IS THE SIZE OF THE TEST SET
	    # IF IT IS NOT DEFINED, SET END OF OPTIMISING SET TO START OF TEST SET
	    start_test = len(self.output)-test_size
	    if end_train == None:
		end_train = start_test

	    # IF NUMBER OF RUNS IS NOT DEFINED, SET IT TO MAXIMUM POSSIBLE (THE NUMBER OF COMPOUNDS BEING OPTIMISED OVER)
	    if number_runs == None:
		number_runs = start_test - training_size

	    print training_size, " initial training compounds"
	    print test_size, " compounds in fixed test set"	
	    print number_runs, " compounds out of", start_test - training_size, " will be used for optimisation"

	    # CREATE MODEL FOR SELECTING THE NEXT COMPOUND
	    modopt = model.Model(n_kers=2, Xtrain=[self.descriptors[:training_size],self.smiles[:training_size]],Xtest=[self.descriptors[training_size:end_train],self.smiles[training_size:end_train]], Ytrain=self.output[:training_size], Ytest=self.output[training_size:end_train], kernel=ker) 

	    # CREATE MODEL TO MAKE PREDICTIONS ON TEST SET USING UPDATED TRAINING SET
	    modtest = model.Model(n_kers=2, Xtrain=modopt.Xtrain, Xtest=[self.descriptors[start_test:],self.smiles[start_test:]], Ytrain=modopt.Ytrain, Ytest=self.output[start_test:], kernel=ker) 

	    # PERFORM REGRESSION ON TEST SET USING INITIAL TRAINING SET
	    modtest.kernel.noise_var = noise
	    modtest.hyperparameters(print_vals=False)
	    regtest = modtest.regression()
	    print "Results on test set using initial training set"
#	    regtest.plot_by_index()

	    # KEEP TEST SET R SQUARED VALUES IN A LIST, THE FIRST VALUE IS THE R SQUARED VALUE FOR PREDICTIONS ON THE TEST SET USING THE ORIGINAL TRAINING SET BEFORE BAYESIAN OPTIMISATION
	    r_sq = []
	    r_sq.append(regtest.r_squared())
	    print "Initial r squared",regtest.r_squared()
	    
	    # BEGIN BAYESIAN OPTIMISATION
	    for i in xrange(number_runs):
		# EVERY 10 RUNS, PRINT THE NUMBER OF THE RUN THE MODEL IS ON CURRENTLY
		if i%10 == 0:
		    print "Iteration",i,"..."
		# SET ACQUISITION FUNCTION
		modopt.acq_func = acquisition_function

		# RUN OPTIMISATION FUNCTION TO FIND THE NEW INPUT VALUE, AUTOMATICALLY ADDED TO THE TRAINING SET AND REMOVED FROM THE OPTIMISATION SET
		newx = modopt.optimization()  

		# RESET TRAINING SET OF THE TEST SET MODEL
		modtest.Xtrain = modopt.Xtrain
		modtest.Ytrain = modopt.Ytrain 
	        modtest.kernel.noise_var = noise # IS THIS NEEDED

		# RESET THE HYPERPARAMETERS OF THE TEST SET MODEL AS THE TRAINING SET HAS CHANGED
		modtest.hyperparameters(print_vals=False)

		# PERFORM REGRESSION ON THE TEST SET USING THE UPDATED TRAINING SET
		regtest = modtest.regression()

		# PRINT R SQUARED FOR EACH RUN IF IT IS REQUESTED
                if i != number_runs-1 and print_interim == True:
			print "r squared",regtest.r_squared()

		# PRINT R SQUARED FOR FINAL RUN
		elif i == number_runs-1:
			print "Results using final training set"
#			regtest.plot_by_index()			
			print "r squared",regtest.r_squared()

		r_sq.append(regtest.r_squared())

	    # ARRAY STARTING AT -1 FOR THE RUN BEFORE OPTIMISATION BEGINS
	    run = np.linspace(-1,number_runs-1,num=number_runs+1)

	    # SAVE R SQUARED FOR EACH RUN IN A TEXT FILE
	    np.savetxt("/home/nafisa/Dropbox/DPhil/Gaussian_processes/results/rsq_" + acquisition_function.abbreviation + "_tr" + str(training_size) + "_te" + str(test_size) + "_runs" + str(number_runs) + ".txt", np.c_[run,r_sq], fmt='%i	%f')
		
	    return modopt,modtest,r_sq
	    
	def q_squared(self, training_size, ker, acquisition_function, noise):
		# Take e.g. 10 molecules
		# Use acquisition function to select next molecule
		# calculate q2
		# 
		# Repeat from step 2
		# 

#	    mod = model.Model(n_kers=2, kernel=ker, Xtrain=[self.descriptors[:training_size],self.smiles[:training_size]],Xtest=[self.descriptors[training_size:],self.smiles[training_size:]], Ytrain=self.output[:training_size], Ytest=self.output[training_size:])
#	    mod.kernel.noise_var = noise
#  	    mod.hyperparameters(print_vals=False)
#	    mod.acq_func = acquisition_function
#  	    print len(mod.Xtrain[1])
#	    newx, newobs = mod.optimization()
#  	    print len(mod.Xtrain[1])
	    
	    cv = cross_validation.Cross_Validation(self.output[:training_size], descs=self.descriptors[:training_size], smiles=self.smiles[:training_size], n_folds=training_size)
	    q_sq, observed, predicted = cv.perform_cv(cv.y, ker, cv.random_folds, q2=True, descs=cv.descs, smiles=cv.smiles)
	    return q_sq, observed, predicted  #r_sq

	    
	
