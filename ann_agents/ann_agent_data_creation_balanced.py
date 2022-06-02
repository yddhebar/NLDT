import numpy as np
import gym
import pickle
from ann_agents import ann_agent_classes_funcs as agents
from rl_envs.custom_envs import CarFollowing
from rl_envs.custom_envs import MyCartPole
import os

#env_name = 'CartPole'
env_name = 'CarFollowing'
#extra_string = '_uniform_lead_acc_2vels'
extra_string = '_uniform_lead_acc_3vels'
agent_name = env_name + extra_string

n_data_pts_per_class = 5000

if env_name == 'CartPole':
    env = MyCartPole()
elif env_name == 'CarFollowing':
    a_range = np.array([-1,0,1])
    env = CarFollowing(lead_profile='uniform', a_range = a_range)
#env = env.unwrapped  #...for behind-the-scenes dynamics of the environment
N_STATES = env.observation_space.shape[0]
n_actions = env.action_space.n

n_iters = 1000
max_data_points = n_actions * n_data_pts_per_class


my_agent = agents.MyAgent(env = env, NET_TYPE='big')
weight_file_name = os.path.join(agent_name + '.pth')
my_agent.set_weights(weight_file_name)

performance_array_list = []
for _ in range(n_actions):
    performance_array_list.append(np.zeros((n_data_pts_per_class, N_STATES + 1)))

action_space = env.action_space

counter_list = np.zeros(n_actions)

while True:
    s = env.reset()
    for i in range(n_iters):
        #env.render()
        a = my_agent.choose_action(s)
        performance_array_list[int(a)][int(counter_list[int(a)]), 0:N_STATES] = s
        performance_array_list[int(a)][int(counter_list[int(a)]),N_STATES:] = a
        if counter_list[a] < n_data_pts_per_class - 1:
            counter_list[a] += 1
        if sum(counter_list) + n_actions >= max_data_points:
            break
        print(f'iter = {i}, counter array = {counter_list}')
        s_, r, done, info = env.step(a)
        s = s_

        if done:
            break

    if sum(counter_list) + n_actions >= max_data_points:
        break


#...SAVING DATA....
performance_array = np.concatenate(performance_array_list, axis = 0)
f_name = agent_name + '_data_' + str(max_data_points) + '_balanced'
print(f'saving data with name: {f_name}')
file_name = os.path.join('..','datasets', f_name + '.data')
np.savetxt(file_name,performance_array, delimiter=',')

env.close()
