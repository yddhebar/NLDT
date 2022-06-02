import os
import pandas as pd
import numpy as np
from iai_multiclass_classification.iai_utils import decision_tree_funcs as dt_funcs
import pickle

max_depth = 3
min_data_pts = 10
tau_accuracy = 0.0

n_links = 10#...for Planar Manipulator problem....
if n_links <= 5:
    torque_max = 200 * n_links  # ...for Planar Manipulator problem....
else:
    torque_max = 400 * n_links

n_features = 2*n_links
n_actions = 3

#...Loading Tree...
#data_name = 'DS1' + '_modified'
#data_name = 'LunarLander-v2_10000' + '_balanced_data'
#data_name = 'LunarLander-v2_tau10_10000'# + '_balanced_data'
#data_name = 'LunarLander-v2_10000_balanced_data'
#data_name = 'ford_data_new_merged'
#data_name = 'Acrobot-v1_10000'# + '_balanced'
#data_name = 'PlanarManipulator_10000'# + '_balanced_data'
data_name = 'PlanarManipulator_' + str(n_links) + '_' + str(torque_max) + '_10000' + '_balanced_data'
#data_name = 'PlanarManipulator_10_2000_10000' + '_balanced_data'
#data_name = 'CartPole-v0_10000'
#data_name = 'Ford1DEnv_10000'
#data_name = 'MountainCar'  + '_balanced'
#extra_name = '_depth10_3terms_report_binary_cons_ratio_dt'

tree_id = 0
extra_name = '_dt_' + str(tree_id)
extra_name = '_classical' + extra_name
#extra_name = '_rga' + '_simple' + extra_name
#tree_name = 'iris_dt'#_pruned'
tree_name = data_name + extra_name#_pruned'
#tree_name = 'LunarLander_balanced_data_dt_0'


my_tree_loc = os.path.join('..','results',data_name,tree_name + '.p')
my_tree = pickle.load(open(my_tree_loc, 'rb'))

#..loading Data...
data_file = os.path.join('..','..','datasets', data_name + '.data')
my_data = pd.read_csv(data_file, header=None)


#..main code....
features = my_data.iloc[:,:n_features]
action_array = my_data.iloc[:,-1]
print('Data Loaded')

training_data_X = np.array(features)
normalized_features = my_tree.normalize_features(training_data_X)

pruned_tree = dt_funcs.pruned_tree_for_depth(my_tree, max_depth)

pruned_tree = dt_funcs.prune_tree_for_accuracy(pruned_tree,
                                               features = normalized_features,
                                               class_labels = action_array,
                                               tau_accuracy=tau_accuracy,
                                               min_data_pts=min_data_pts)

pruned_tree.get_tree_max_depth()
pruned_tree.count_number_of_active_nodes()
pruned_tree.compute_eqn_stats()

print('Saving Tree with name ', tree_name + '_pruned.p''')
file_name = os.path.join('..','results', data_name, tree_name + '_pruned.p')
pickle.dump(pruned_tree,open(file_name,'wb'))

