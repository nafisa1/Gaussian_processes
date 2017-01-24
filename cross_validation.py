import numpy as np
import model

def order(x, y):
	y1 = y
	y_sorted = sorted(y1)
	x_sorted = [x for y,x in sorted(zip(y,x))]
	return x_sorted, y_sorted

def get_test_set(x, y, fraction_test):
    step = int(round(1/fraction_test))
    x_test_set = np.array(x[::step])
    y_test_set = np.array(y[::step])
    if isinstance(x,list):
        del x[::step]
        cv_x = x    
    elif isinstance(x,np.ndarray):
        cv_x = []
        position=0
        for index, row in enumerate(x):
            if index != position:
                cv_x.append(row)
            else:
                position += step
	cv_x = np.asarray(cv_x)
    if isinstance(y,list):
        del y[::step]
        cv_y = y    
    elif isinstance(y,np.ndarray):
        cv_y = []
        position=0
        for index, row in enumerate(y):
            if index != position:
                cv_y.append(row)
            else:
                position += step
    cv_y = np.asarray(cv_y)
    return x_test_set, y_test_set, cv_x, cv_y

def get_stratified_folds(cv_x, cv_y, n_folds=10):
    x_all_folds = []
    y_all_folds = []
    for fold_number in xrange(n_folds):
        x_fold = []
	y_fold = []
	
	if isinstance(cv_x,np.ndarray):
		position = fold_number
		for number in xrange(cv_x.shape[0]/n_folds):
			x_fold.append(cv_x[position])    
            		position += n_folds	
	elif isinstance(cv_x, list):
		position = fold_number
		for number in xrange(len(cv_x)/n_folds):
			x_fold.append(cv_x[position])    
            		position += n_folds
        for number in xrange(cv_y.shape[0]/n_folds):
		position = fold_number
		y_fold.append(cv_y[position])    
		position += n_folds
        x_all_folds.append(x_fold)
	y_all_folds.append(y_fold)
        
    x_validation_sets = []
    x_training_sets = []
    y_validation_sets = []
    y_training_sets = []
    for i in xrange(n_folds):
        x_training = []
	y_training = []
        for fold in xrange(n_folds):
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

def get_binned_folds(cv_data_x, cv_data_y, threshold, n_folds=10):
    active_x = []
    inactive_x = []
    active_y = []
    inactive_y = []
      
    for i,number in enumerate(cv_data_y):
        if number > threshold:
            active_y.append(number)
            active_x.append(cv_data_x[i])
        
        else:
            inactive_y.append(number)
            inactive_x.append(cv_data_x[i])
    print "active x:", len(active_x), "inactive x:", len(inactive_x), "active y:", len(active_y), "inactive y:", len(inactive_y)
  
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
    active_step = len(shuf_active_y)//n_folds
    x_inactive_folds = []
    y_inactive_folds = []
    inactive_step = len(shuf_inactive_y)//n_folds

    for i in xrange(n_folds):
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
    
    for i,value in enumerate(shuf_active_x[active_step*n_folds:]):
        x_active_folds[i].append(value)

    for i,value in enumerate(shuf_active_y[active_step*n_folds:]):
        y_active_folds[i].append(value)

    for i,value in enumerate(shuf_inactive_x[inactive_step*n_folds:]):
        x_inactive_folds[i].append(value)

    for i,value in enumerate(shuf_inactive_y[inactive_step*n_folds:]):
        y_inactive_folds[i].append(value)

    x_folds = []
    y_folds = []
    for i in xrange(n_folds):
        x_folds.append(x_active_folds[i]+x_inactive_folds[i])
        y_folds.append(y_active_folds[i]+y_inactive_folds[i])

    x_validation_sets = []
    x_training_sets = []
    y_validation_sets = []
    y_training_sets = []

    for i in xrange(n_folds):
        x_training = []
        y_training = []
        for fold in xrange(n_folds):
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

def perform_cv(kern, x_validation_sets, x_training_sets, y_validation_sets, y_training_sets):
    n_folds = y_validation_sets.shape[0]
    r_sq = []

    for i in xrange(n_folds):
        run = model.Model(y_training_sets[i], y_validation_sets[i], smiles_train=x_training_sets[i], smiles_test=x_validation_sets[i], kernel=kern)
        run_regression = run.regression()
        r_sq.append(run_regression.r_squared())
    print r_sq
    return np.mean(r_sq) # modified to return mean instead of lists

def repeated_CV(kern, cv_data_x, cv_data_y, iterations, threshold, nfolds=10)
	means = []
	for i in xrange(iterations):
	    x_validation_sets, x_training_sets, y_validation_sets, y_training_sets = get_binned_folds(cv_data_x, cv_data_y, threshold, n_folds=nfolds)
	    r_sq = perform_cv(kern, x_validation_sets, x_training_sets, y_validation_sets, y_training_sets)
	    means.append(r_sq)
	return means, np.mean(means)
