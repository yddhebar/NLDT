import os
from iai_utils import visualize_tree
import pandas as pd
import numpy as np
from iai_utils import decision_tree_funcs as dt_funcs
import pickle
import time

n_links = 6
if n_links <= 5:
    torque_max = 200 * n_links  # ...for Planar Manipulator problem....
else:
    torque_max = 500 * n_links

n_features = 4#2*n_links
n_actions = 3
tree_id = 0
fine_tuned_simulations = False

#...Loading Tree...
#data_name = 'DS1' + '_modified'
#env_name = 'LunarLander-v2'
#env_name = 'Ford1DEnv'
#env_name = 'Acrobot-v1'
#data_name = 'LunarLander-v2_10000' + '_balanced_data'
#data_name = 'LunarLander-v2_tau10_10000'# + '_balanced_data'
#data_name = 'LunarLander-v2_q_values_confident_00_10000'
#data_name = 'Acrobot-v1_10000'# + '_balanced'
#data_name = 'PlanarManipulator_10000' + '_balanced_data'
#data_name = 'PlanarManipulator_5_10000' + '_balanced_data'
#data_name = 'PlanarManipulator_' + str(n_links) + '_' + str(torque_max) + '_10000' + '_balanced_data'
#data_name = 'ford_data_new_merged'
#data_name = 'CartPole-v0_10000'
#data_name = 'Ford1DEnv_10000'
#data_name = 'MountainCar'# + '_balanced'
data_name = 'iris'
#data_name = 'DS3_matlab'
#testing_data_name = 'MountainCar_testing'
#testing_data_name = 'iris'
#testing_data_name = 'LunarLander-v2_testing_10000'
#testing_data_name = 'Acrobot-v1_testing_10000'
#testing_data_name = 'PlanarManipulator_testing_10000'
#testing_data_name = 'PlanarManipulator_' + str(5) + '_testing_10000'
#testing_data_name = 'PlanarManipulator_' + str(n_links) + '_' + str(torque_max) + '_testing_10000'
testing_data_name = data_name
#testing_data_name = 'CartPole-v0_testing_10000'
#testing_data_name = 'Ford1DEnv_testing_10000'
extra_name = '_dt_' + str(tree_id)
#extra_name = '_classical' + extra_name
#extra_name = '_classical' + '_simple' + extra_name
#extra_name = '_rga' + '_simple' + extra_name
#extra_name = '_rga' + extra_name
tree_name = data_name + extra_name + '_pruned'
#tree_name = 'LunarLander_balanced_data_dt_0' + '_pruned'
#tree_name = env_name + '_fine_tuned_from_env_depth_2'# + '_' + str(tree_id)
#tree_name = env_name + '_fine_tuned_from_data_depth_6' + '_' + str(tree_id)

env_name = 'PlanarManipulator' + '_' + str(n_links) + '_' + str(torque_max)

if fine_tuned_simulations:
    tree_name = env_name + '_fine_tuned_from_env_depth_3' + extra_name
tree_name = 'iris_classical_dt_0'
my_tree_loc = os.path.join('..','results', data_name,tree_name + '.p')
my_tree = pickle.load(open(my_tree_loc, 'rb'))

w_array = my_tree.get_weights_from_tree()
my_tree.assign_weight_to_nodes_from_array(w_array)

#..loading Data...
data_file = os.path.join('..','datasets', data_name + '.data')
my_data = pd.read_csv(data_file, header=None)

testing_data_file = os.path.join('..','datasets', testing_data_name + '.data')#data_file
testing_data = pd.read_csv(testing_data_file, header = None)


#...Data preparation...
features = my_data.iloc[:,:n_features]
action_array = my_data.iloc[:,-1]

testing_features = testing_data.iloc[:,:n_features]
testing_action_array = testing_data.iloc[:,-1]
print('Data Loaded')

training_data_X = np.array(features)
testing_data_X = np.array(testing_features)

training_data_X = my_tree.normalize_features(training_data_X)
testing_data_X = my_tree.normalize_features(testing_data_X)

#..main code....

t1_ =  time.time()
my_tree.data_distribution_in_tree_from_c_label(features= training_data_X,
                                               class_array = action_array)

accuracy = dt_funcs.compute_accuracy_from_c_label(my_tree, training_data_X, action_array)
testing_accuracy = dt_funcs.compute_accuracy_from_c_label(my_tree, testing_data_X, testing_action_array)

print('prediction accuracy = ',accuracy)
print('Testing Accuracy = %.2f'% testing_accuracy)

#..Print Tree Properties...
my_tree.get_tree_max_depth()
my_tree.count_number_of_active_nodes()
my_tree.compute_eqn_stats()
print('Max Depth of Tree = %d' % my_tree.tree['max_depth'])
print('Total Nodes = %d' % my_tree.tree['total_nodes'])
print('Total Active Nodes = %d' % my_tree.tree['n_active_nodes'])

my_tree.write_text_to_tree(detailed_text = True,
                           colored_active_nodes = True)

file_name = os.path.join('..','..', 'results', 'tree_plots','tree_plot')

file_name_table = os.path.join('..','..', 'results', 'tree_plots', 'table_latex.tex')

visualize_tree.plot(filename=file_name, tree = my_tree.tree)
my_tree.write_tree_table_latex(file_name_table)

#...Latex Format...
print('Result in Latex Format')
print('Tr Accu \t Test Acu \t Depth  \t total nodes \t # Rules \t Avg Rule Len')
print('%.2f & %.2f & %d & %d & %d & %.2f\\\\\\hline' % (accuracy,
                                                  testing_accuracy,
                                                  my_tree.tree['max_depth'],
                                                  my_tree.tree['total_nodes'],
                                                  my_tree.tree['n_active_nodes'],
                                                        my_tree.tree['eqn_len_avg']))

t2_ = time.time()

print('It look this much time: %f' % (t2_ - t1_))