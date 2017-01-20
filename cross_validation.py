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
    cv_x = np.asarray(cv_x)
    cv_y = np.asarray(cv_y)
    return x_test_set, y_test_set, cv_x, cv_y

def get_stratified_folds(cv_data, n_folds=10):
    all_folds = []
    for fold_number in xrange(n_folds):
        position = fold_number
        fold = []
        for number in xrange(cv_data.shape[0]/n_folds):
            fold.append(cv_data[position])    
            position += n_folds
        all_folds.append(fold)
        
    validation_sets = []
    training_sets = []
    for i in xrange(n_folds):
        training = []
        for fold in xrange(n_folds):
            if i == fold:
                validation_sets.append(all_folds[fold])
            else:
                training.append(all_folds[fold])
        training_sets.append(training)
    validation_sets = np.asarray(validation_sets)
    training_sets = np.asarray(training_sets)
    return validation_sets, training_sets

def perform_cv(kern, x_validation_sets, x_training_sets, y_validation_sets, y_training_sets):
    n_folds = x_validation_sets.shape[0]
    r_sq = []
    for i in xrange(n_folds):
        run = model.Model(y_training_sets[i], y_validation_sets[i], smiles_train=x_training_sets[i], smiles_test=x_validation_sets[i], kernel=kern)
        run_regression = run.regression()
        run_regression.plot_by_index()
        r_sq.append(run_regression.r_squared())
    return r_sq #, kern.sig_var, kern.lengthscale, kern.noise_var

