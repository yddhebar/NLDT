import sys
import os
from os import sys, path
import shutil
import numpy as np
import pickle
import pandas as pd
import time
import multiprocessing as mp
sys.path.append(path.join('..','..'))
from iai_utils.iai_dt_classes_and_funcs import Tree as IAI_Tree
import random

from iai_utils import decision_tree_funcs as dt_funcs
from main_codes.parameter_file import GlobalParameters


global_params = GlobalParameters()

#...Global Parameters...
# data_name = 'dtlz2'
#data_name = 'DS1'
# data_name = 'iris'
#data_name = 'CartPole-v0_10000'
#data_name = 'Acrobot-v1_10000'# + '_balanced'
#data_name = 'Ford1DEnv_10000'
#data_name = 'MountainCar' + '_balanced'#_q_values'
#data_name = 'LunarLander-v2_50000'# + '_balanced_data'
#data_name = 'LunarLander-v2_q_values_confident_00_10000'# + '_balanced_data'
#data_name = 'DS3_matlab'
# data_name = 'cardio_dataset_10class'
#data_name = 'PlanarManipulator_10000' + '_balanced_data'
#data_name = 'PlanarManipulator_5_10000' + '_balanced_data'
data_name = 'PlanarManipulator_10_' + str(2000) + '_10000'+ '_balanced_data'

n_features = 20
max_depth = 3
n_classes = 3

extra_name = '_' + global_params.ll_algorithm + '_v2'#'_rga'#'_classical'#'_' + global_params.impurity_measure
tree_name = data_name + extra_name
training_data_share = 1.0

my_tree = IAI_Tree()
min_size = 10
tau_impurity = 0.05
n_runs = 10
#class_dict = {0:0,1:1}#,2:2,3:3}
class_dict = {}
for c in range(n_classes):
    class_dict.update({c:c})

results_dir = os.path.join('..','results', data_name)

#..load data...
data_file = os.path.join('..', '..', 'datasets', data_name + '.data')
#data_file = os.path.join('..', '..', 'iai_journal_problems', data_name + '.data')

my_data = pd.read_csv(data_file, header = None)

features = my_data.iloc[:, :n_features]
action_array = my_data.iloc[:, -1]
print('Data Loaded')

training_data_X = np.array(features)
training_data_Y = action_array
#..Done with data-loading...

#..make the parent dicectory...
try:
    os.mkdir(results_dir)
except:
    print('%s directory already exits!!!' % results_dir)

#..store whole data in the result_dir...
shutil.copyfile(data_file, os.path.join(results_dir, data_name + '.data'))


def run_training(run, training_data_X = training_data_X):
    print('run = ', run)
    final_agent_name = tree_name + '_dt_' + str(run) + '.p'
    file_name = os.path.join(results_dir, final_agent_name)
    if os.path.exists(file_name):
        print('Result Exists for Run: %d' % run)
        #return

    random.seed(run)
    np.random.seed(run)
    t_i = time.perf_counter()
    my_tree.train_tree(features = training_data_X,
                       class_labels= action_array,
                       max_depth= max_depth,
                       min_size= min_size,
                       tau_impurity= tau_impurity,
                       class_dict = class_dict,
                       training_data_share = training_data_share)
    t_f = time.perf_counter()

    print('total number of nodes = ', my_tree.tree['total_nodes'])

    print('Saving Tree with name ', final_agent_name)

    pickle.dump(my_tree,open(file_name,'wb'))

    training_data_X = my_tree.normalize_features(training_data_X)

    train_ids = my_tree.tree['data_ids']['train_ids']
    test_ids = my_tree.tree['data_ids']['test_ids']

    training_accuracy = dt_funcs.compute_accuracy_from_c_label(my_tree,
                                                               training_data_X[train_ids,:],
                                                               action_array.iloc[train_ids])

    testing_accuracy = dt_funcs.compute_accuracy_from_c_label(my_tree,
                                                               training_data_X[test_ids, :],
                                                               action_array.iloc[test_ids])

    print('prediction accuracy')
    print('Training = %.2f' % training_accuracy)
    print('Testing = %.2f' % testing_accuracy)
    print('done with decision tree training')
    print('Time Taken = %.2fs'%(t_f - t_i))
    return t_f - t_i


if __name__ == '__main__':
    t1 = time.perf_counter()
    n_cpus = mp.cpu_count()
    print('N processors: ', n_cpus)

    with mp.Pool(processes=n_cpus) as pool:
        results = pool.map(run_training, list(range(n_runs)))
    #print(results)

    t2 = time.perf_counter()
    print(f'Finished in {t2 - t1} seconds')
    print('Done Training all Runs')
    res_array = np.array(results)
    print('Time Taken = %.2fs \\pm %.2f' % (np.mean(res_array), np.std(res_array)))
