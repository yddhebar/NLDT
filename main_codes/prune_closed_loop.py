#...here will prune those segments of NLDT which are not operational in closed loop...
import os
import gym
from iai_utils import visualize_tree
import pandas as pd
import numpy as np
from iai_utils import decision_tree_funcs as dt_funcs
import pickle
from rl_envs import custom_envs
import time

n_links = 10
if n_links <= 5:
    torque_max = 200 * n_links  # ...for Planar Manipulator problem....
else:
    torque_max = 400 * n_links

fine_tuned_simulations = True
class_reassignment = True
pruning_flag = True
store_data = True
n_features = 2*n_links
n_actions = 3
tree_id = 0
n_time_steps = 10000
max_depth = 3
min_data_pts = 10
tau_accuracy = 0.0

env_name = 'PlanarManipulator' + '_' + str(n_links) + '_' + str(torque_max)
#...Loading Tree...
#env_name = 'LunarLander-v2'
#env_name = 'Ford1DEnv'
#env_name = 'Acrobot-v1'
#data_name = 'LunarLander-v2_10000' + '_balanced_data'
#data_name = 'LunarLander-v2_q_values_confident_00_10000'
#data_name = 'Acrobot-v1_10000'# + '_balanced'
data_name = 'PlanarManipulator_' + str(n_links) + '_' + str(torque_max) + '_10000' + '_balanced_data'
#data_name = 'PlanarManipulator_10_2000_10000' + '_balanced_data'
#data_name = 'CartPole-v0_10000'
#data_name = 'Ford1DEnv_10000'
#data_name = 'MountainCar' + '_balanced'
#data_name = 'iris'
#data_name = 'DS3_matlab'
extra_name = '_dt_' + str(tree_id)
extra_name = '_classical' + extra_name
#extra_name = '_rga' + extra_name
if fine_tuned_simulations:
    tree_name = env_name + '_fine_tuned_from_env_depth_3' + extra_name
else:
    tree_name = data_name + extra_name + '_pruned'

# creating environment
if env_name == 'Ford1DEnv':
    env = custom_envs.Ford1DEnv(lead_profile='uniform')#, init_vel=0)
if 'PlanarManipulator' in env_name:
    env = custom_envs.PlanarSerialManipulator(n_links=n_links, torque_value=torque_max)
else:
    env = gym.make(env_name)

my_tree_loc = os.path.join('..','results', data_name,tree_name + '.p')
my_tree = pickle.load(open(my_tree_loc, 'rb'))

w_array = my_tree.get_weights_from_tree()
my_tree.assign_weight_to_nodes_from_array(w_array)

#..loading Data...
t1_ = time.time()
try:
    data_file = os.path.join('..','results', data_name, tree_name + '.data')
    my_data = np.loadtxt(data_file, delimiter=',')
    states_array = my_data[:,:n_features]
    action_array = my_data[:,-1]
except:
    print('Preparing the data.....')
    #...Data preparation...
    states_array = np.ones([n_time_steps, n_features])*np.nan
    action_array = np.ones(n_time_steps)*np.nan
    t = 0
    while t < n_time_steps:
        s = env.reset()
        for i in range(n_time_steps):
            states_array[t,:] = s
            action = int(my_tree.choose_action(s))
            action_array[t] = int(action)
            s, r_, done, info = env.step(action)
            t = t + 1
            if t >= n_time_steps:
                break
            if done:
                break

    #...Saving the data...
    print('Saving the data with name: %s' % (tree_name + '.data'))
    my_data = np.hstack((states_array, action_array.reshape([n_time_steps,1])))
    np.savetxt(data_file, X=my_data, delimiter=',')


normalized_states_array = my_tree.normalize_features(states_array)
my_tree.data_distribution_in_tree_from_c_label(features= normalized_states_array,
                                               class_array = pd.Series(action_array))
if class_reassignment:
    my_tree.reassign_classes()

#....Closed Loop Pruning...
if pruning_flag:
    my_tree = dt_funcs.pruned_tree_for_depth(my_tree, max_depth)

    my_tree = dt_funcs.prune_tree_for_accuracy(my_tree,
                                                   features = normalized_states_array,
                                                   class_labels = pd.Series(action_array),
                                                   tau_accuracy=tau_accuracy,
                                                   min_data_pts=min_data_pts)


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
print('Tr Accu  \t Depth  \t total nodes \t # Rules \t Avg Rule Len')
print(' %d & %d & %d & %.2f\\\\\\hline' % (my_tree.tree['max_depth'],
                                           my_tree.tree['total_nodes'],
                                           my_tree.tree['n_active_nodes'],
                                           my_tree.tree['eqn_len_avg']))

t2_ = time.time()

print('Saving Tree with name ', tree_name + '_new_class.p''')
file_name = os.path.join('..','results', data_name, tree_name + '_new_class.p')
pickle.dump(my_tree,open(file_name,'wb'))

print('It look this much time: %f' % (t2_ - t1_))