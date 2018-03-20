import numpy as np
import model # CIRCULAR IMPORT
import utils

class Cross_Validation(object):

	def __init__(self, y, fraction_test=0.2, n_folds=10, n_kers=1, descs=None, smiles=None, threshold=None):
		self.y = y
		self.fraction_test = fraction_test
		self.n_folds = n_folds
		self.n_kers = n_kers
		self.descs = descs
		self.smiles = smiles
		self.threshold = threshold	

#		change_vars = raw_input("Would you like to change values of the default latin hypercube sampling variables? Enter y or n: ")
			
#		if change_vars == "y":
			
#			n_parameters = input("Enter the number of hyperparameters, not including noise variance: ")

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

	def random_folds(self, array, scrambled_indices=None):
	    indices = np.arange(len(array))
	    if scrambled_indices is None:
        	scrambled_indices = np.random.permutation(indices)
	    rearranged_array = array[scrambled_indices]
	    splits = np.array_split(rearranged_array, self.n_folds)
    
	    validation_sets = []
	    training_sets = []
	    for i in xrange(self.n_folds):
        	training = []
	        for fold_number in xrange(self.n_folds):
        	    if i == fold_number:
        	        validation_sets.append(splits[fold_number])
        	    else:
        	        training.append(splits[fold_number])
		if len(splits[0].shape) == 1:
		        training_sets.append(np.hstack(training))
		elif len(splits[0].shape) == 2:
		        training_sets.append(np.vstack(training))


	    validation_sets = np.asarray(validation_sets)
	    training_sets = np.asarray(training_sets)

	    return validation_sets, training_sets, scrambled_indices


	def stratified_folds(self, array):
	    all_folds = []
	    for fold_number in xrange(self.n_folds):
	        fold = []
	        position = fold_number
	        for number in xrange(array.shape[0]/self.n_folds):
	            fold.append(array[position])    
	            position += self.n_folds
	        all_folds.append(fold)
    	
	    if len(array) % n_folds != 0:
	        for i,value in enumerate(array[(array.shape[0] - array.shape[0]%n_folds):]):
	            all_folds[i].append(value)     

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
	
	    return validation_sets, training_sets, None
	
	def binned_folds(self, array, iteration=0):
	    active = []
	    inactive = []

	    for i,number in enumerate(array):
	        if number > self.threshold:
	            active.append(number)
        
	        else:
	            inactive.append(number)

	    if iteration == 0:
		    print "active:", len(active), "inactive:", len(inactive)
  
	    p = np.random.permutation(len(active))
	    shuf_active = (np.asarray(active))[p]
	
	    q = np.random.permutation(len(inactive))
	    shuf_inactive = (np.asarray(inactive))[q]
	
	    active_folds = np.array_split(shuf_active,self.n_folds)
	    inactive_folds = np.array_split(shuf_inactive,self.n_folds)
	
	    all_folds = []
	    for i in xrange(self.n_folds):
	        all_folds.append(np.concatenate((active_folds[i],inactive_folds[i])))
	
	    validation_sets = []
	    training_sets = []
	    	
	    for i in xrange(self.n_folds):
	        training = []
	        for j,fold in enumerate(all_folds):#fold in xrange(self.n_folds):
	            if i == j:
	                validation_sets.append(fold)
	            else:
	                training.append(fold)
	        training = [item for sublist in training for item in sublist]
	        training_sets.append(training)
	        
	    validation_sets = np.asarray(validation_sets)
	    indices = np.concatenate((p,q))
	    
	    return validation_sets, training_sets, indices

	def perform_cv(self, y, kern, fold_type, q2=False, descs=None, smiles=None):
	    r_sq = []
#	    fold_type=self.random_folds
	    y_val_sets, y_tr_sets, indices = fold_type(y)

	    if descs is not None:
	        desc_val_sets, desc_tr_sets, indices = fold_type(descs, scrambled_indices=indices)
	    if smiles is not None:
	        smiles_val_sets, smiles_tr_sets, indices = fold_type(np.array(smiles), scrambled_indices=indices)

	    observed = []
	    predicted = []
			
	    for i in xrange(self.n_folds):
			if descs is None:
				run = model.Model(Ytrain=y_tr_sets[i], Ytest=y_val_sets[i], Xtrain=np.ndarray.tolist(smiles_tr_sets[i]), Xtest=np.ndarray.tolist(smiles_val_sets[i]), kernel=kern, threshold=self.threshold)
			if smiles is None:
				run = model.Model(Ytrain=y_tr_sets[i], Ytest=y_val_sets[i], Xtrain=desc_tr_sets[i], Xtest=desc_val_sets[i], kernel=kern, threshold=self.threshold)
			else:
				run = model.Model(Ytrain=y_tr_sets[i], Ytest=y_val_sets[i], Xtrain=[desc_tr_sets[i],np.ndarray.tolist(smiles_tr_sets[i])], Xtest=[desc_val_sets[i],np.ndarray.tolist(smiles_val_sets[i])], kernel=kern, threshold=self.threshold)
			run.hyperparameters(print_vals=False)
			run_regression = run.regression()
			print run_regression.post_mean, ",", run_regression.Ytest, ",", i
			r_sq.append(run_regression.r_squared())
			observed.append(run_regression.Ytest)
			predicted.append(run_regression.post_mean)

            if q2==True:
                return r_sq#observed, predicted
#		else:
#			return r_sq, indices

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

