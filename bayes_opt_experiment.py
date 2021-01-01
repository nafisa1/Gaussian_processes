#!/usr/bin/python
import kernels
import acquisition
import run_experiment
import dataextraction

filename = 'datasets/EVOTEC_OC_DATASET_20170328.xlsx'
evotecIDs = 'EVOTEC_COMPOUND_ID'
output_name = 'ACTIVITY_1_IC50_NM_AVG'
smiles_name = 'CD_SMILES'
descriptor_names = ['CD_MOLWEIGHT','TPSA']#'LOGD','LOGP','TPSA']


example = run_experiment.Experiment()
extractor = dataextraction.Extract()

# EXTRACT DATA

smiles, pic50s, names, descriptors = extractor.get_data(filename, evotecIDs, smiles_name, output_name, descriptor_names,upper_threshold=20000.0)

training_size = 20
test_size = 80
k = kernels.Composite(kernels.RBF(),kernels.Matern())
n = 0.01
acq_func = acquisition.Random()

total_compounds = len(names)

# SET UP MODELS

modopt,modtest = example.bayes_opt(total_compounds, training_size, test_size, k, acq_func, noise=n, end_train=None, number_runs=5, print_interim=True)

#r = modtest.regression()
#r.plot_all()


