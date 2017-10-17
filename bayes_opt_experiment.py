#!/usr/bin/python
import kernels
import acquisition
import run_experiment

filename = 'datasets/EVOTEC_OT_DATASET_20170328.xlsx'
id_name = 'EVOTEC_ID'#_COMPOUND_ID'
output_name = 'ACTIVITY_1_IC50__UM'#_AVG'
smiles_name = 'CD_SMILES'
descriptor_names = ['CD_MOLWEIGHT','TPSA']#'LOGD','LOGP','TPSA']


example = run_experiment.Experiment()
smiles, pic50s, descriptors = example.get_data(filename, id_name, smiles_name, output_name, descriptor_names,upper_threshold=20000.0)

training_size = 20
test_size = 20
k = kernels.Composite(kernels.RBF(),kernels.Matern())
n = 0.01
acq_func = acquisition.Random()

modopt,modtest, r_sq = example.bayes_opt(training_size, test_size, k, acq_func, noise=n, end_train=None, number_runs=None)
