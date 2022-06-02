import numpy as np
import pickle
import gym
import os

n_data_pts_total = 10000
balanced_data_flag = False
testing_data_flag = False
trajectory_flag = False
q_flag = True#...to indicate if q_values need to be stored OR p_values....

agent = pickle.load(open('mountain_car_agent.p','rb'))

env = gym.make('MountainCar-v0')

action_counter = np.array([0,0,0])

data_array = np.ones([n_data_pts_total, 2 + 3 + 1]) * np.nan  # ....2 states, 3 p-values and one action

n_data_pts = 0
path_completion_array = np.zeros([n_data_pts_total,1])

if balanced_data_flag:
    n_pts_per_action = np.ceil(n_data_pts_total/3)

while True:
    observation = env.reset()
    totalReward = 0
    if n_data_pts >= n_data_pts_total:
        break
    while True:
        print('action dist = [%d, %d, %d], n_data_pts = %d' % (action_counter[0],
                                                               action_counter[1],
                                                               action_counter[2],n_data_pts))
        if totalReward == 0:
            # For step 1, only
            action_probab = agent.start_action_probab(observation, q_flag = q_flag)

        else:
            action_probab = agent.return_action_probab(observation, reward, q_flag = q_flag)

        action = int(action_probab[-1])
        observation_, reward, done, _ = env.step(action)
        path_completion_array[n_data_pts] = done
        totalReward += reward

        if balanced_data_flag:
            if action_counter[int(action)] < n_pts_per_action:
                action_counter[int(action)] += 1
                data_array[n_data_pts, :] = np.concatenate([observation, action_probab])
                n_data_pts += 1
        else:
            action_counter[int(action)] += 1
            data_array[n_data_pts, :] = np.concatenate([observation, action_probab])
            n_data_pts += 1

        if n_data_pts >= n_data_pts_total:
            break

        if done:
            agent.end(reward)
            #episode_rewards[i] = totalReward
            break
        observation = observation_

#....Save The Data...
fname = 'MountainCar'

if q_flag:
    fname += '_q_values'

if trajectory_flag:
    fname += '_trajectory'
    data_array = np.hstack((data_array, path_completion_array))
if balanced_data_flag:
    fname += '_balanced'
if testing_data_flag:
    fname += '_testing'

file_loc = os.path.join('..','..','datasets', fname + '.data')
np.savetxt(fname=file_loc, X=data_array, delimiter=', ')

env.close()