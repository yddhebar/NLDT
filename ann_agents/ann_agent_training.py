import gym
import numpy as np
from ann_agents import ann_agent_classes_funcs as agents
from rl_envs.custom_envs import CarFollowing
from rl_envs.custom_envs import MyCartPole
from rl_envs import custom_envs

#env_name = 'CartPole-v0'
#env_name = 'CartPole'
#env_name = 'MountainCar'
env_name = 'CarFollowing'
#env_name = 'MountainCar-v0'
a_range = np.array([-1,1]) #np.array([-2,-1,0,1])
lead_profile = 'uniform'
agent_extension = 'trail'
if env_name == 'CarFollowing':
    env = CarFollowing(lead_profile=lead_profile,
                    a_range=a_range,)
    agent_extension = ''#'_' + lead_profile + '_lead_acc_' + str(len(a_range)) + 'vels'
elif env_name == 'CartPole':
    env = MyCartPole()
elif env_name == 'MountainCar':
    env = custom_envs.MyMountainCar()
else:
    env = gym.make(env_name)

#...Hyper Parameters...

#agent_extension = ''
n_episodes = 100#400

BATCH_SIZE = 32
LR = 0.01 #....learning rate
EPSILON = 0.9 #...epsilon greedy policy
GAMMA = 0.9 #...discount factor
TARGET_REPLACE_ITER = 100 #.... Target-net update frequency...
MEMORY_CAPACITY = 2000 #....Replay Memory Capacity...

chasing_time_array = np.zeros(n_episodes)
total_reward_array = np.zeros(n_episodes)


dqn = agents.DQN(env=env,
                 BATCH_SIZE=BATCH_SIZE,
                 LR=LR,
                 EPSILON=EPSILON,
                 GAMMA=GAMMA,
                 TARGET_REPLACE_ITER=TARGET_REPLACE_ITER,
                 MEMORY_CAPACITY=MEMORY_CAPACITY,
                 NET_TYPE='big')

print('\nCollecting Experience')
for i_episode in range(n_episodes):
    s = env.reset()
    ep_r = 0#..episode reward...

    while True:

        #env.render()
        a = dqn.choose_action(s)

        s_, r, done, info = env.step(a)

        dqn.store_transition(s, a, r, s_)

        ep_r += r

        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r ', round(ep_r,2))

        if done:
            print('Ep: ', i_episode,
                  '| Ep_r ', round(ep_r, 2))
            break

        s = s_

    total_reward_array[i_episode] = ep_r
    #chasing_time_array[i_episode] = env.time_taken

#....save the trained model...
filename = env_name + agent_extension + '.pth'
print(f'saving the agent with name: {filename}')
dqn.save_model_parameters(filename=filename)

np.savetxt('total_reward_array_' + env_name + agent_extension + '.txt', total_reward_array, delimiter = ' ')
np.savetxt('chasing_time_array_' + env_name + agent_extension + '.txt', chasing_time_array, delimiter= ' ')
