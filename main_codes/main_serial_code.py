import sys
#...This is binary Split NLDT only
import os
import shutil
import numpy as np
import pickle
import pandas as pd
import time

from iai_utils.iai_dt_classes_and_funcs import Tree as IAI_Tree

from iai_utils import decision_tree_funcs as dt_funcs
from main_codes.parameter_file import GlobalParameters

global_params = GlobalParameters()

if __name__ == '__main__':
    n_links = 10#...for Planar Manipulator problem....
    if n_links <= 5:
        torque_max = 200*n_links#...for Planar Manipulator problem....
    else:
        torque_max = 400*n_links
    #...Global Parameters...
    # data_name = 'dtlz2'
    #data_name = 'DS1' + '_modified'
    data_name = 'iris'
    #data_name = 'CartPole-v0_10000'
    #data_name = 'Ford1DEnv_10000'
    #data_name = 'MountainCar'# + '_balanced'#_q_values'
    #data_name = 'LunarLander-v2_q_values_confident_00_10000' + '_balanced_data'
    #data_name = 'LunarLander-v2_10000'
    #data_name = 'LunarLander-v2_10000' + '_balanced_data'
    #data_name = 'LunarLander-v2_tau10_10000'# + '_balanced_data'
    #data_name = 'PlanarManipulator_10000' + '_balanced_data'
    #data_name = 'PlanarManipulator_10_10000' + '_balanced_data'
    #data_name = 'PlanarManipulator_' + str(n_links) + '_' + str(torque_max) + '_10000' + '_balanced_data'
    #data_name = 'DS3_matlab'
    # data_name = 'cardio_dataset_10class'
    #data_name = 'ford_data_new_merged'

    n_features = 4#2*n_links
    max_depth = 4
    n_classes = 3

    extra_name = '_' + global_params.ll_algorithm#'_' + global_params.impurity_measure
    tree_name = data_name + extra_name
    training_data_share = 1.0

    my_tree = IAI_Tree()
    min_size = 5
    tau_impurity = 0.05
    n_runs = 1
    class_dict = {}#{0:0,1:1,2:2,3:3}
    for c in range(n_classes):
        class_dict.update({c:c})
    if 'iris' in data_name:
        class_dict = None


    results_dir = os.path.join('..', 'results', data_name)

    #..load data...
    data_file = os.path.join('..', 'datasets', data_name + '.data')

    my_data = pd.read_csv(data_file, header = None)

    features = my_data.iloc[:, :n_features]
    action_array = my_data.iloc[:, -1]
    print('Data Loaded')

    training_data_X = np.array(features)
    training_data_Y = action_array

    if global_params.impurity_measure == 'weighted_gini':
        q_values = np.array(my_data.iloc[:,n_features:-1])
        data_weights = np.max(q_values,axis = 1)
        #..normalize between 1 and 10
        data_weights = 1 + 9*(data_weights - np.min(data_weights))/(np.max(data_weights) - np.min(data_weights))
    else:
        data_weights = None

    #..Done with data-loading...

    #..make the parent dicectory...
    try:
        os.mkdir(results_dir)
    except:
        print('%s directory already exits!!!' % results_dir)

    #..store whole data in the result_dir...
    shutil.copyfile(data_file, os.path.join(results_dir, data_name + '.data'))

    t1 = time.perf_counter()
    for run in range(n_runs):
        print('run = ', run)
        my_tree.train_tree(features = training_data_X,
                           class_labels= action_array,
                           max_depth= max_depth,
                           min_size= min_size,
                           tau_impurity= tau_impurity,
                           class_dict = class_dict,
                           training_data_share = training_data_share,
                           data_weights=data_weights)

        print('total number of nodes = ', my_tree.tree['total_nodes'])

        final_agent_name = tree_name + '_dt_' + str(run) + '.p'
        print('Saving Tree with name ', final_agent_name)
        file_name = os.path.join(results_dir, final_agent_name)
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

    t2 = time.perf_counter()
    print(f'Finished in {t2 - t1} seconds')
    print('Done Training all Runs')
