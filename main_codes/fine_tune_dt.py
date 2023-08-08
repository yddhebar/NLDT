import numpy as np
#....Nonlinear Decision Tree for Imitation Learning...
import gym
import os
import pandas as pd
import pickle
import sys
sys.path.append(os.path.join('..','..'))
from iai_utils.fine_tuning_from_experience import fine_tune_dt as fine_tune_from_env
from iai_utils.fine_tuning_from_data import fine_tune_dt as fine_tune_from_data
import time
from rl_envs.custom_envs import Ford1DEnv
from rl_envs.custom_envs import PlanarSerialManipulator

#env_name = 'LunarLander-v2'
#env_name = "Ford1DEnv"
#env_name = 'Acrobot-v1'
n_links = 10
if n_links <= 5:
    torque_max = 200 * n_links  # ...for Planar Manipulator problem....
else:
    torque_max = 400 * n_links
env_name = 'PlanarManipulator' + '_' + str(n_links) + '_' + str(torque_max)

n_features = 2*n_links
fine_tune_mode = 'from_env'#'from_data'#
tree_id = 0

#data_name = 'MountainCar'#_q_values'#_balanced'
#data_name = 'LunarLander-v2_10000' + '_balanced_data'
#data_name = 'Ford1DEnv_10000'
#data_name = 'Acrobot-v1_10000'# + '_balanced'
#data_name = 'PlanarManipulator_10000' + '_balanced_data'
#data_name = 'PlanarManipulator_5_10000' + '_balanced_data'
data_name = 'PlanarManipulator_' + str(n_links) + '_' + str(torque_max) + '_10000' + '_balanced_data'
#extra_name = '_weighted_gini' + '_dt_0'
extra_name = '_dt_' + str(tree_id)
extra_name = '_classical' + extra_name
#extra_name = '_classical_simple' + extra_name
#extra_name = '_rga_simple' + extra_name
#extra_name = '_rga' + extra_name
tree_name = data_name + extra_name + '_pruned'

agent_file = os.path.join('..','results', data_name, tree_name + '.p')
data_file = os.path.join('..','..','datasets', data_name + '.data')
#my_data = pickle.load(open(data_file, 'rb'))
my_data = pd.read_csv(data_file, header=None)
my_DT = pickle.load(open(agent_file, 'rb'))

#...Initialize Variables...
# creating environment
if env_name == 'Ford1DEnv':
    env = Ford1DEnv(lead_profile='uniform')#, init_vel=0)
elif 'PlanarManipulator' in env_name:
    env = PlanarSerialManipulator(n_links=n_links, torque_value=torque_max)
else:
    env = gym.make(env_name)

my_DT.get_tree_max_depth()
depth = my_DT.tree['max_depth']
print('Max Depth = %d' % depth)
agent_file_name = env_name + '_fine_tuned_' + fine_tune_mode + '_depth_' + str(depth) + extra_name + '.p'

X_data = my_data.iloc[:,:n_features]
Y_data = my_data.iloc[:,-1]
t_0 = time.time()
if fine_tune_mode == 'from_env':
    my_DT_fine_tuned = fine_tune_from_env(my_DT, env, p_explore=0.1)
elif fine_tune_mode == 'from_data':
    X_data = np.array(X_data)
    my_DT_fine_tuned = fine_tune_from_data(my_DT, X_data, Y_data, 0)
t_f = time.time()

#..Writing tree for Latex..
#my_DT_fine_tuned.write_text_to_tree()

agent_file_loc = os.path.join('..','results', data_name, agent_file_name)
print('Saving Tree with name ', agent_file_name)
pickle.dump(my_DT_fine_tuned,open(agent_file_loc,'wb'))
print('Time Taken = %.2f' % (t_f - t_0))

