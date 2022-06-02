import numpy as np
import gym
import pandas as pd
import pickle
from ann_agents import ann_agent_classes_funcs as agents
from rl_envs.custom_envs import CarFollowing
from rl_envs.custom_envs import MyCartPole
import os

p_values_flag = True#..to indicate if p_values are supposed to be stored....
testing_flag = True#...is the created dataset used for testing?
#env_name = 'CartPole'
env_name = 'CarFollowing'
a_range = np.array([-1,1])

extra_string = ''#'_uniform_lead_acc_' + str(len(a_range)) + 'vels'
#extra_string = '_uniform_lead_acc_3vels'
#extra_string = '_uniform_lead_acc_2vels'
#extra_string = '_discrete_lead_acc_2vels'
#extra_string = ''

agent_name = env_name + extra_string

n_data_pts_per_class = 5000

if env_name == 'CartPole':
    env = MyCartPole()
    n_classes = 2
elif env_name == 'CarFollowing':
    env = CarFollowing(lead_profile='uniform', a_range = a_range)
    n_classes = len(a_range)
#env = env.unwrapped  #...for behind-the-scenes dynamics of the environment
N_STATES = env.observation_space.shape[0]
N_ACTIONS = env.action_space.n

n_iters = 1000
max_data_points = n_classes * n_data_pts_per_class

my_agent = agents.MyAgent(env = env, NET_TYPE='big')
weight_file_name = os.path.join(agent_name + '.pth')
my_agent.set_weights(weight_file_name)

if p_values_flag:
    performance_array = np.zeros((max_data_points, N_STATES + N_ACTIONS + 1))
else:
    performance_array = np.zeros((max_data_points, N_STATES + 1))
counter = 0

while True:
    s = env.reset()
    ep_reward = 0
    for i in range(n_iters):
        #env.render()
        if p_values_flag:
            action_array = my_agent.get_action_array(s)
            a = int(np.max(action_array))
            performance_array[counter, N_STATES:] = action_array
        else:
            a = my_agent.choose_action(s)
            performance_array[counter, N_STATES:] = a
        performance_array[counter, :N_STATES] = s

        counter += 1
        if counter >= max_data_points:
            break
        print('iter = ', i)
        s_, r, done, info = env.step(a)
        s = s_
        ep_reward += r
        if done:
            print('Ep_reward = %d' % ep_reward)
            break

    if counter >= max_data_points:
        break


#...SAVING DATA....
if testing_flag:
    f_name = agent_name + '_testing' + '_' + str(max_data_points) + '.data'
else:
    f_name = agent_name + '_' + str(max_data_points) + '.data'
print(f'Saving file with name: {f_name}')
file_name = os.path.join('..','datasets', f_name)
np.savetxt(file_name,performance_array, delimiter=',')

#...Class Statistics..
action_array = performance_array[:,-1]
unique_actions = np.unique(action_array)
class_distribution = -1*np.ones(len(unique_actions))

for i in range(len(unique_actions)):
    n_a = sum(action_array == unique_actions[i])
    class_distribution[i] = n_a

print(f'Class Distribution: {class_distribution}')

env.close()
