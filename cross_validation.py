import numpy as np
import model

def get_folds_and_test_set(data, n_folds, fraction_test):
    number_of_samples = data.shape[0]
    number_for_cv = int((number_of_samples*(1-fraction_test))//n_folds)*n_folds
    cv_data = data[:number_for_cv]
    test_set = data[number_for_cv:]
    
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
                val_set = all_folds[fold]
            else:
                training.append(all_folds[fold])
        validation_sets.append(val_set)
        training_sets.append(training)
    validation_sets = np.asarray(validation_sets)
    training_sets = np.asarray(training_sets)
    return validation_sets, training_sets, test_set


def perform_cv(kern, x_validation_sets, x_training_sets, y_validation_sets, y_training_sets):
    n_folds = x_validation_sets.shape[0]
    r_sq = []
    for i in xrange(n_folds):
        run = model.Model(y_training_sets[i], y_validation_sets[i], smiles_train=x_training_sets[i], smiles_test=x_validation_sets[i], kernel=kern)
        run_regression = run.regression()
        run_regression.plot_by_index()
        r_sq.append(run_regression.r_squared())
    print max(r_sq), kern.sig_var, kern.lengthscale, kern.noise_var
