import numpy as np
import utils
import model
import cross_validation

class Experiment(object):

  def __init__(self, smiles, pic50s, names, descriptors, total_compounds, training_size, test_size, ker, acquisition_function, noise=0.01, number_runs=None, print_interim=False):

    self.smiles = smiles
    self.pic50s = pic50s
    self.names = names
    self.descriptors = descriptors
    self.total_compounds = total_compounds
    self.training_size = training_size
    self.test_size = test_size
    self.ker = ker
    self.acquisition_function = acquisition_function
    self.noise = noise
    self.number_runs = number_runs
    self.print_interim = print_interim
    
    self.start_test = total_compounds-test_size

    
    # IF NUMBER OF RUNS IS NOT DEFINED, SET IT TO MAXIMUM POSSIBLE (THE NUMBER OF COMPOUNDS BEING OPTIMISED OVER)
    if number_runs == None:
      self.number_runs = self.start_test - training_size
      
    print training_size, " initial training compounds"
    print test_size, " compounds in fixed test set"	
    print number_runs, " compounds out of", start_test - training_size, " will be used for optimisation"
    
    self.modopt = model.Model(n_kers=2, Xtrain=[descriptors[:training_size],smiles[:training_size]],Xtest=[descriptors[training_size:start_test],smiles[training_size:start_test]], Ytrain=output[:training_size], Ytest=output[training_size:start_test], kernel=ker)
    self.modopt.kernel.noise_var = noise
    self.modopt.hyperparameters(print_vals=True)
 
    # SET ACQUISITION FUNCTION   
    self.modopt.acq_func = acquisition_function
#    modopt.mse_hyperparameters(print_vals=False)  

    print "initial hps",self.modopt.kernel.kers[0].sig_var, self.modopt.kernel.kers[0].lengthscale, self.modopt.kernel.kers[1].sig_var, self.modopt.kernel.kers[1].lengthscale
   
    # CREATE MODEL TO MAKE PREDICTIONS ON TEST SET USING UPDATED TRAINING SET
    self.modtest = model.Model(n_kers=2, Xtrain=self.modopt.Xtrain, Xtest=[descriptors[-test_size:],smiles[-test_size:]], Ytrain=self.modopt.Ytrain, Ytest=output[-test_size:], kernel=self.modopt.kernel) ## changed kernel from opt to train 


  def bayes_opt(self):
  
#    p = np.random.permutation(len(self.smiles))
#    self.smiles, self.output, self.descriptors = [self.smiles[i] for i in p], self.output[p], [self.descriptors[i] for i in p]

    # POINT WHERE TEST SET STARTS IS DEFINED AS THE LAST x COMPOUNDS WHERE x IS THE SIZE OF THE TEST SET
    # IF IT IS NOT DEFINED, SET END OF OPTIMISING SET TO START OF TEST SET

    # CREATE MODEL FOR SELECTING THE NEXT COMPOUND
    modopt = model.Model(n_kers=2, Xtrain=[self.descriptors[:training_size],self.smiles[:training_size]],Xtest=[self.descriptors[training_size:start_test],self.smiles[training_size:start_test]], Ytrain=self.output[:training_size], Ytest=self.output[training_size:start_test], kernel=ker)
    modopt.kernel.noise_var = noise
    modopt.hyperparameters(print_vals=True)
 
    # SET ACQUISITION FUNCTION   
    modopt.acq_func = acquisition_function
#    modopt.mse_hyperparameters(print_vals=False)  

    print "initial hps",modopt.kernel.kers[0].sig_var, modopt.kernel.kers[0].lengthscale, modopt.kernel.kers[1].sig_var, modopt.kernel.kers[1].lengthscale
   
    # CREATE MODEL TO MAKE PREDICTIONS ON TEST SET USING UPDATED TRAINING SET
    modtest = model.Model(n_kers=2, Xtrain=modopt.Xtrain, Xtest=[self.descriptors[start_test:],self.smiles[start_test:]], Ytrain=modopt.Ytrain, Ytest=self.output[start_test:], kernel=modopt.kernel) ## changed kernel from opt to train 
    
    # PERFORM REGRESSION ON TEST SET USING INITIAL TRAINING SET
    regtest = modtest.regression()
    print "Results on test set using initial training set"
    
    # KEEP TEST SET R SQUARED VALUES IN A LIST, THE FIRST VALUE IS THE R SQUARED VALUE FOR PREDICTIONS ON THE TEST SET USING THE ORIGINAL TRAINING SET BEFORE BAYESIAN OPTIMISATION 
    r_sq = []
    r_sq.append(regtest.r_squared())
    print "Initial r squared",regtest.r_squared()
   
    # BEGIN BAYESIAN OPTIMISATION
    for i in xrange(number_runs): 

      # EVERY 10 RUNS, PRINT THE NUMBER OF THE RUN THE MODEL IS ON CURRENTLY
      if i%10 == 0:
        print "Iteration",i,"..."
        
      # RUN OPTIMISATION FUNCTION TO FIND THE NEW INPUT VALUE, AUTOMATICALLY ADDED TO THE TRAINING SET AND REMOVED FROM THE OPTIMISATION SET
#######################?????????????????????
  
      newx,newy = modopt.optimization()
      
      # RESET TRAINING SET OF THE TEST SET MODEL
      modtest.Xtrain = modopt.Xtrain
      modtest.Ytrain = modopt.Ytrain
        
      
      # RESET THE HYPERPARAMETERS OF THE TEST SET MODEL AS THE TRAINING SET HAS CHANGED
  #    modopt.kernel.noise_var = noise
#      modopt.mse_hyperparameters(print_vals=False)

      # PERFORM REGRESSION ON THE TEST SET USING THE UPDATED TRAINING SET
      regtest = modtest.regression()
      r_sq.append(regtest.r_squared())
      
      # PRINT R SQUARED FOR EACH RUN IF IT IS REQUESTED
      if i != number_runs-1 and print_interim == True:
        print "r squared",regtest.r_squared()
        
      # PRINT R SQUARED FOR FINAL RUN
      elif i == number_runs-1:
        print "Results using final training set"
        print "r squared",regtest.r_squared()
        print "mse",regtest.mse()
        

    print "training set mse: ", modopt.regression().mse()
    print "test set mse: ", modtest.regression().mse()
    modtest.regression().plot_by_index()
#    modopt.regression().plot_by_index()
        
    # ARRAY STARTING AT 0 FOR THE RUN BEFORE OPTIMISATION BEGINS
    run = np.linspace(0,number_runs,num=number_runs+1)
      
    # SAVE R SQUARED FOR EACH RUN IN A TEXT FILE
    np.savetxt("/home/nafisa/Dropbox/DPhil/Gaussian_processes/results/rsq_" + acquisition_function.abbreviation + "_tr" + str(training_size) + "_te" + str(test_size) + "_runs" + str(number_runs) + ".txt", np.c_[run,r_sq], fmt='%i	%f')
      
    return modopt,modtest

  def bayes_opt2(self,training_size, test_size, ker, acquisition_function, noise=0.01, number_runs=None, end_train=None, print_interim=False):
    
    
    # BEGIN BAYESIAN OPTIMISATION
    for i in xrange(number_runs):
      # EVERY 10 RUNS, PRINT THE NUMBER OF THE RUN THE MODEL IS ON CURRENTLY
      if i%10 == 0:
        print "Iteration",i,"..."
        
      # RUN OPTIMISATION FUNCTION TO FIND THE NEW INPUT VALUE, AUTOMATICALLY ADDED TO THE TRAINING SET AND REMOVED FROM THE OPTIMISATION SET
      modopt.hyperparameters(print_vals=False) #######################?????????????????????
      print modopt.kernel.kers[0].sig_var, modopt.kernel.kers[0].lengthscale, modopt.kernel.kers[1].sig_var, modopt.kernel.kers[1].lengthscale ### at i=0 this should be the same as modtest hparams above
      newx = modopt.optimization()
      
      # RESET TRAINING SET OF THE TEST SET MODEL
      modtest.Xtrain = modopt.Xtrain
      modtest.Ytrain = modopt.Ytrain
      
      # RESET THE HYPERPARAMETERS OF THE TEST SET MODEL AS THE TRAINING SET HAS CHANGED
      modtest.hyperparameters(print_vals=False)
      
      # PERFORM REGRESSION ON THE TEST SET USING THE UPDATED TRAINING SET
      regtest = modtest.regression()
      r_sq.append(regtest.r_squared())
      
      # PRINT R SQUARED FOR EACH RUN IF IT IS REQUESTED
      if i != number_runs-1 and print_interim == True:
        print "r squared",regtest.r_squared()
        
      # PRINT R SQUARED FOR FINAL RUN
      elif i == number_runs-1:
        print "Results using final training set"
        print "r squared",regtest.r_squared()
        
    # ARRAY STARTING AT 0 FOR THE RUN BEFORE OPTIMISATION BEGINS
    run = np.linspace(0,number_runs,num=number_runs+1)
      
    # SAVE R SQUARED FOR EACH RUN IN A TEXT FILE
    np.savetxt("/home/nafisa/Dropbox/DPhil/Gaussian_processes/results/rsq_" + acquisition_function.abbreviation + "_tr" + str(training_size) + "_te" + str(test_size) + "_runs" + str(number_runs) + ".txt", np.c_[run,r_sq], fmt='%i	%f')
      
    return modopt,modtest,r_sq
	    
  def q_squared(self, training_size, ker, acquisition_function):
		# Take e.g. 10 molecules
		# Use acquisition function to select next molecule
		# calculate q2
		# Repeat from step 2
		   
	    cv = cross_validation.Cross_Validation(self.output[:training_size], descs=self.descriptors[:training_size], smiles=self.smiles[:training_size], n_folds=training_size)
	    q_sq, observed, predicted = cv.perform_cv(cv.y, ker, cv.random_folds, q2=True, descs=cv.descs, smiles=cv.smiles)
	    return q_sq, observed, predicted  #r_sq

	    
	
