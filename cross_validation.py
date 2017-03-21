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
		self.hparameter_choices = utils.LHS().combinations

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
	    return self.x_test_set, self.y_test_set, self.cv_x, self.cv_y		

	def get_stratified_folds(self):
	    x_all_folds = []
	    y_all_folds = []
	    for fold_number in xrange(self.n_folds):
	        x_fold = []
		y_fold = []
		
		if isinstance(self.cv_x,np.ndarray):
			position = fold_number
			for number in xrange(self.cv_x.shape[0]/self.n_folds):
				x_fold.append(self.cv_x[position])    
	            		position += self.n_folds	
		elif isinstance(self.cv_x, list):
			position = fold_number
			for number in xrange(len(self.cv_x)/self.n_folds):
				x_fold.append(self.cv_x[position])    
	            		position += self.n_folds
	        for number in xrange(self.cv_y.shape[0]/self.n_folds):
			position = fold_number
			y_fold.append(self.cv_y[position])    
			position += self.n_folds
	        x_all_folds.append(x_fold)
		y_all_folds.append(y_fold)
        
	    x_validation_sets = []
	    x_training_sets = []
	    y_validation_sets = []
	    y_training_sets = []
	    for i in xrange(self.n_folds):
	        x_training = []
		y_training = []
	        for fold in xrange(self.n_folds):
	            if i == fold:
	                x_validation_sets.append(x_all_folds[fold])
			y_validation_sets.append(y_all_folds[fold])
	            else:
	                x_training.append(x_all_folds[fold])
			y_training.append(y_all_folds[fold])
	        x_training_sets.append(x_training)
		y_training_sets.append(y_training)
	
	    x_validation_sets = np.asarray(x_validation_sets)
	    x_training_sets = np.asarray(x_training_sets)
	    y_validation_sets = np.asarray(y_validation_sets)
	    y_training_sets = np.asarray(y_training_sets)
	
	    return x_validation_sets, x_training_sets, y_validation_sets, y_training_sets
	
	def get_binned_folds(self, iteration=0):
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
	    shuf_active_x = [active_x[i] for i in p]
	    shuf_active_y = [active_y[i] for i in p]
	
	    q = np.random.permutation(len(inactive_y))
	    shuf_inactive_x = [inactive_x[i] for i in q]
	    shuf_inactive_y = [inactive_y[i] for i in q]
	
	    act_position=0
	    inact_position=0
	
	    x_active_folds = []
	    y_active_folds = []
	    active_step = len(shuf_active_y)//self.n_folds
	    x_inactive_folds = []
	    y_inactive_folds = []
	    inactive_step = len(shuf_inactive_y)//self.n_folds

	    for i in xrange(self.n_folds):
	        x_act_fold = shuf_active_x[act_position:act_position+active_step]
	        y_act_fold = shuf_active_y[act_position:act_position+active_step]
	        act_position += active_step
	        x_active_folds.append(x_act_fold) 
	        y_active_folds.append(y_act_fold)
	        x_inact_fold = shuf_inactive_x[inact_position:inact_position+inactive_step]
	        y_inact_fold = shuf_inactive_y[inact_position:inact_position+inactive_step]
	        inact_position += inactive_step
	        x_inactive_folds.append(x_inact_fold) 
	        y_inactive_folds.append(y_inact_fold) 
	    
	    for i,value in enumerate(shuf_active_x[active_step*self.n_folds:]):
	        x_active_folds[i].append(value)
	
	    for i,value in enumerate(shuf_active_y[active_step*self.n_folds:]):
	        y_active_folds[i].append(value)
	
	    for i,value in enumerate(shuf_inactive_x[inactive_step*self.n_folds:]):
	        x_inactive_folds[i].append(value)
	
	    for i,value in enumerate(shuf_inactive_y[inactive_step*self.n_folds:]):
	        y_inactive_folds[i].append(value)
	
	    x_folds = []
	    y_folds = []
	    for i in xrange(self.n_folds):
	        x_folds.append(x_active_folds[i]+x_inactive_folds[i])
	        y_folds.append(y_active_folds[i]+y_inactive_folds[i])
	
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

	def perform_cv(self, kern, x_validation_sets, x_training_sets, y_validation_sets, 	y_training_sets):
	    r_sq = []
	
	    for i in xrange(self.n_folds):
	        run = model.Model(Ytrain=y_training_sets[i], Ytest=y_validation_sets[i], smiles_train=x_training_sets[i], smiles_test=x_validation_sets[i], kernel=kern, threshold=self.threshold)
	        run_regression = run.regression()
	        r_sq.append(run_regression.r_squared())
	    return np.mean(r_sq)

	def repeated_CV(self, default_kern, hparams, iterations=10, lhs_kern=None):
		
		iteration_means = []
		for i in xrange(iterations):
			iteration_mean = []
			x_validation_sets, x_training_sets, y_validation_sets, y_training_sets = self.get_binned_folds(iteration=i) # ALLOW OPTION FOR BINNED OR STRATIFIED
			print "Iteration ", i
			default_r_sq = self.perform_cv(default_kern, x_validation_sets, x_training_sets, y_validation_sets, y_training_sets)
			iteration_mean.append(default_r_sq)
			    
			for j in xrange(len(hparams)):
				# if isinstance(self.kernel, kernels.Composite): 
				# for i in xrange(self.n_kers):
				# lhs_kern.kers[i].lengthscale=
				# lhs_kern.kers[i].sig_var=
				# else:
				lhs_kern.noise_var=self.hparameter_choices[j][2]
	    			r_sq = self.perform_cv(lhs_kern, x_validation_sets, x_training_sets, y_validation_sets, y_training_sets)
				iteration_mean.append(r_sq)
		    	iteration_means.append(iteration_mean)	

		means = (np.asarray(iteration_means)).T
		self.means_over_iters = np.mean(means, axis=1)

		index = np.argmax(self.means_over_iters)
		if index == 0:
			best_noise = default_kern.noise_var
		else:
			best_noise = hparams[index-1][2] # corrected?

		return best_noise, means, self.means_over_iters	

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

