import numpy as np
import model # CIRCULAR IMPORT
import utils

class Cross_Validation(object):

	def __init__(self, x, y, fraction_test=0.2, n_folds=10, n_kers=1, threshold=None):
		self.x = x
		self.y = y
		self.fraction_test = fraction_test
		self.n_folds = n_folds
		self.n_kers = n_kers
		self.threshold = threshold	

#		change_vars = raw_input("Would you like to change values of the default latin hypercube sampling variables? Enter y or n: ")
			
#		if change_vars == "y":
			
#			n_parameters = input("Enter the number of hyperparameters, not including noise variance: ")
#			n_samples = input("Enter the number of samples from the hyperparameter space: ")
#			lowb = input("Enter the lower bounds for the hyperparameters as a list [(lengthscale, signal variance)*number of hyperparameters, noise variance]: ")
#			upb = input("Enter the upper bounds for the hyperparameters as a list [(lengthscale, signal variance)*number of hyperparameters, noise variance]: ")
#			divs = input("Enter the number of evenly-spaced samples for each hyperparameter, in one list: ")
#			self.hparameter_choices = utils.LHS(parameters=n_parameters, n_choices=n_samples, lower=lowb, upper=upb, divisions=divs).combinations

#		elif change_vars == "n":
#			print("All LHS variables will remain as default.")

	def order(self):
		y = self.y
		self.x = [self.x for self.y,self.x in sorted(zip(self.y,self.x))]
		self.y = sorted(y)

	def get_test_set(self):
	    step = int(round(1/self.fraction_test))
	    self.x_test_set = np.array(self.x[4::step])
	    self.y_test_set = np.array(self.y[4::step])
	    if isinstance(self.x,list):
	        del self.x[4::step]
	        self.cv_x = self.x    
	    elif isinstance(self.x,np.ndarray):
	        cv_x = []
	        position=4
	        for index, row in enumerate(self.x):
	            if index != position:
	                cv_x.append(row)
	            else:
	                position += step
		self.cv_x = np.asarray(cv_x)
	    if isinstance(self.y,list):
	        del self.y[4::step]
	        cv_y = self.y    
	    elif isinstance(self.y,np.ndarray):
	        cv_y = []
	        position=4
	        for index, row in enumerate(self.y):
	            if index != position:
	                cv_y.append(row)
	            else:
	                position += step
	    self.cv_y = np.asarray(cv_y)
	    print len(self.x_test_set), self.y_test_set.shape, len(self.cv_x), self.cv_y.shape
	    return self.x_test_set, self.y_test_set, self.cv_x, self.cv_y		

	def random_folds(self, array, folds, scrambled_indices=None):
	    indices = np.arange(len(array))
	    if scrambled_indices == None:
        	scrambled_indices = np.random.permutation(indices)
	    rearranged_array = array[scrambled_indices]
	    splits = np.array_split(rearranged_array, folds)
    
	    validation_sets = []
	    training_sets = []
	    for i in xrange(folds):
        	training = []
	        for fold_number in xrange(folds):
        	    if i == fold_number:
        	        validation_sets.append(splits[fold_number])
        	    else:
        	        training.append(splits[fold_number])
        	training_sets.append(training)

	    validation_sets = np.asarray(validation_sets)
	    training_sets = np.asarray(training_sets)

	    return validation_sets, training_sets, scrambled_indices


	def stratified_folds(self, array):
	    all_folds = []
	    for fold_number in xrange(self.n_folds):
	        fold = []
		
	        for number in xrange(self.array.shape[0]/self.n_folds):
			position = fold_number
			fold.append(self.array[position])    
			position += self.n_folds
	        all_folds.append(fold)
        
	    validation_sets = []
	    training_sets = []
	    for i in xrange(self.n_folds):
	        training = []
	        for fold in xrange(self.n_folds):
	            if i == fold:
	                validation_sets.append(all_folds[fold])
	            else:
	                training.append(all_folds[fold])
	        training_sets.append(training)
	
	    validation_sets = np.asarray(validation_sets)
	    training_sets = np.asarray(training_sets)
	
	    return validation_sets, training_sets
	
	def binned_folds(self, iteration=0):
	    active_x = []
	    inactive_x = []
	    active_y = []
	    inactive_y = []

	    for i,number in enumerate(self.cv_y):
	        if number > self.threshold:
	            active_y.append(number)
	            active_x.append(self.cv_x[i])
        
	        else:
	            inactive_y.append(number)
	            inactive_x.append(self.cv_x[i])

	    if iteration == 0:
		    print "active x:", len(active_x), "inactive x:", len(inactive_x), "active y:", len	(active_y), "inactive y:", len(inactive_y)
  
	    p = np.random.permutation(len(active_y))
	    shuf_active_x = (np.asarray(active_x))[p]#[active_x[i] for i in p]
	    shuf_active_y = (np.asarray(active_y))[p]
	
	    q = np.random.permutation(len(inactive_y))
	    shuf_inactive_x = (np.asarray(inactive_x))[q]
	    shuf_inactive_y = (np.asarray(inactive_y))[q]
	
	    x_active_folds = np.array_split(shuf_active_x,self.n_folds)
	    y_active_folds = np.array_split(shuf_active_y,self.n_folds)
	    x_inactive_folds = np.array_split(shuf_inactive_x,self.n_folds)
	    y_inactive_folds = np.array_split(shuf_inactive_y,self.n_folds)
	
	    x_folds = []
	    y_folds = []
	    for i in xrange(self.n_folds):
	        x_folds.append(np.concatenate((x_active_folds[i],x_inactive_folds[i])))
	        y_folds.append(np.concatenate((y_active_folds[i],y_inactive_folds[i])))
	    print x_folds
	
	    x_validation_sets = []
	    x_training_sets = []
	    y_validation_sets = []
	    y_training_sets = []
	
	    for i in xrange(self.n_folds):
	        x_training = []
	        y_training = []
	        for fold in xrange(self.n_folds):
	            if i == fold:
	                x_validation_sets.append(x_folds[fold])
	                y_validation_sets.append(np.asarray(y_folds[fold]))
	            else:
	                x_training.append(x_folds[fold])
	                y_training.append(y_folds[fold])
	        x_training = [item for sublist in x_training for item in sublist]
	        y_training = np.array([item for sublist in y_training for item in sublist])
	        x_training_sets.append(x_training)
	        y_training_sets.append(y_training)
	        
	    y_validation_sets = np.asarray(y_validation_sets)
	    y_training_sets = np.asarray(y_training_sets)
	    
	    return x_validation_sets, x_training_sets, y_validation_sets, y_training_sets

	def perform_cv(self, y, fold_type, kern, descs=None, smiles=None):
	    r_sq = []

	    y_val_sets, y_tr_sets, si = fold_type(y, folds)

	    if descs is not None:
	        desc_val_sets, desc_tr_sets, si = fold_type(descs, folds, si)
	    if smiles is not None:
	        smiles_val_sets, smiles_tr_sets, si = fold_type(np.array(smiles), folds, si)

	
	    for i in xrange(self.n_folds):
		if descs is None:
			run = model.Model(Ytrain=y_training_sets[i], Ytest=y_validation_sets[i], Xtrain=np.ndarray.tolist(smiles_training_sets[i]), Xtest=np.ndarray.tolist(smiles_validation_sets[i]), kernel=kern, threshold=self.threshold)
		if smiles is None:
			run = model.Model(Ytrain=y_training_sets[i], Ytest=y_validation_sets[i], Xtrain=desc_training_sets[i], Xtest=desc_validation_sets[i], kernel=kern, threshold=self.threshold)
		else:
	        	run = model.Model(Ytrain=y_training_sets[i], Ytest=y_validation_sets[i], Xtrain=[desc_training_sets[i],np.ndarray.tolist(smiles_training_sets[i])], Xtest=[desc_validation_sets[i],np.ndarray.tolist(smiles_validation_sets[i])], kernel=kern, threshold=self.threshold)
		run.hyperparameters()
	        run_regression = run.regression()
	        r_sq.append(run_regression.r_squared())
	    return r_sq, si

	def repeated_CV(self, kern, y, descs=None, smiles=None, iterations=10, lhs_kern=None):
		
		iteration_means = []
		for i in xrange(iterations):
			print "Iteration ", i
			r_sq = self.perform_cv(y, kern, descs=descs, smiles=smiles)
			iteration_means.append(r_sq)
			    

		means = (np.asarray(iteration_means)).T
		self.means_over_iters = np.mean(means, axis=1)

		#index = np.argmax(self.means_over_iters)

		return means, self.means_over_iters	

	def test_set_results(self, test_kern):
		index = np.argmax(self.means_over_iters)
		best = self.hparameter_choices[index-1] # corrected?
		print best
		test_kern.noise_var=best[2]

		test_run = model.Model(self.cv_y, self.y_test_set, smiles_test=self.x_test_set, smiles_train=self.cv_x, kernel=test_kern, threshold=self.threshold)
		test_regression = test_run.regression()  
		test_regression.plot_by_index()
		r_sq = test_regression.r_squared()
		print r_sq
		Ytr_mean = test_run.Ytrain_mean
		predictions = test_regression.post_mean + Ytr_mean
		 
		utils.classif(predictions, self.y_test_set, self.threshold)
		# post_mean, upper, lower = test_regression.predict()

