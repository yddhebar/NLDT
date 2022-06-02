#...this code if for the agent created using the PPO algorithm...
import gym
from ann_agents.PPO import PPO, Memory
from PIL import Image
import torch
from rl_envs.custom_envs import CarFollowing
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


env_name = 'CarFollowing'
env = CarFollowing(lead_profile='uniform', init_vel=20)
n_episodes = 1
max_timesteps = 200
p_flag = 1#..store p values..

filename = "PPO_{}.pth".format(env_name)
directory = ''  # "./preTrained/"

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

n_latent_var = 64           # number of variables in hidden layer
lr = 0.0007
betas = (0.9, 0.999)
gamma = 0.99                # discount factor
K_epochs = 4                # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
#############################################

memory = Memory()
ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)

ppo.policy_old.load_state_dict(torch.load(directory + filename))

ep_rewards_array = np.zeros(n_episodes)
state_array = np.zeros([max_timesteps + 1, state_dim])

for ep in range(1, n_episodes + 1):
    ep_reward = 0
    state = env.reset()
    state_array[0,:] = state
    for i in range(max_timesteps):
        action_array = ppo.policy_old.act_test(state, memory, p_flag=p_flag)
        action = int(action_array[-1])

        state, reward, done, _ = env.step(action)
        state_array[i+1,:] = state
        ep_reward += reward
        if done:
            break

    ep_rewards_array[ep-1] = ep_reward

print('Ep Rewards Avg +- Std = %.2f \\pm %.2f' % (np.mean(ep_rewards_array),
                                                  np.std(ep_rewards_array)))

print('Plot States....')

#....
time_array = np.arange(i + 1)
state_array_trancated = state_array[0:i+1,:]

plt.plot(time_array, state_array_trancated[:,0], 'r-')
plt.xlabel('Time Step')
plt.ylabel('Rel Dist')
plt.title('Rel Dist Plot.')
plt.grid()
plt.show()


