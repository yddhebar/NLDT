import sys
import os
sys.path.append(os.path.join('..'))
import gym
from ann_agents.PPO import PPO, Memory
from PIL import Image
import torch
from rl_envs.custom_envs import MyCartPole
import numpy as np
import time
import multiprocessing as mp

render = True
if render:
    n_cpus = 1
else:
    n_cpus = np.min([50, mp.cpu_count()])
############## Hyperparameters ##############
n_partitions = 50
n_episodes_per_partition = 100
n_episodes = n_partitions*n_episodes_per_partition
max_timesteps = 200
time_delay = 0.0#.. in seconds...
# env_name = "MyCartPole"
# env_name = "MountainCar-v0"
env_name = 'CartPole-v0'
#env_name = 'LunarLander-v2'
#env_name = 'Acrobot-v1'

# creating environment
if env_name == 'MyCartPole':
    env = MyCartPole()  # gym.make(env_name)
else:
    env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

n_latent_var = 64  # number of variables in hidden layer
lr = 0.0007
betas = (0.9, 0.999)
gamma = 0.99  # discount factor
K_epochs = 4  # update policy for K epochs
eps_clip = 0.2  # clip parameter for PPO
#############################################

save_gif = False

filename = "PPO_{}.pth".format(env_name)
directory = ''  # "./preTrained/"

memory = Memory()
ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)

ppo.policy_old.load_state_dict(torch.load(directory + filename))
print('Parameter Setting Done......')


def run_episode(ep):
    ep_reward = 0
    success_ = 0
    landing_ = 0
    state = env.reset()
    for t in range(max_timesteps):
        action = ppo.policy_old.act(state, memory)
        state, reward, done, _ = env.step(action)
        ep_reward += reward
        if render:
            env.render()
            time.sleep(time_delay)
        if save_gif:
            img = env.render(mode='rgb_array')
            img = Image.fromarray(img)
            img.save('./gif/{}.jpg'.format(t))
        if done:
            if 'LunarLander' in env_name:
                if not env.env.lander.awake: #r_ == 100:
                    print('Yes')
                    success_ = 1
                else:
                    print('No')

                if (state[-1] == 1 and state[-2] == 1) or reward == 100:
                    landing_ = 1
                    print('But Yes')
                else:
                    #env.render()
                    print(state)
                    print('Here')
            break
    print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
    return np.array([ep_reward, success_, landing_])


if __name__ == '__main__':
    t_0 = time.time()
    with mp.Pool(processes=n_cpus) as pool:
        results = pool.map(run_episode, list(range(n_episodes)))

    t_f = time.time()
    env.close()
    rewards_array = np.array(results)

    results = np.array(results)
    rewards_array = results[:, 0]
    success_rate_array = results[:, 1]
    landing_array = results[:, 2]

    print('Success Rate = %d' % np.sum(success_rate_array))
    print('Landing Rate = %d' % np.sum(landing_array))
    print('# Episodes with Ep_r >= 200 = %d' % (np.sum(rewards_array >= 200)))
    print('# Episodes with Ep_r >= 150 = %d' % (np.sum(rewards_array >= 150)))
    print('# Episodes with Ep_r >= 100 = %d' % (np.sum(rewards_array >= 100)))
    print('# Episodes with Ep_r >= 50 = %d' % (np.sum(rewards_array >= 50)))
    print('# Episodes with Ep_r >= 0 = %d' % (np.sum(rewards_array >= 0)))
    print('Rewards Stats')
    print('Min \t Max \t Avg \t Std')
    print('%.2f \t %.2f \t %.2f \t %.2f' % (np.min(rewards_array),
                                            np.max(rewards_array),
                                            np.mean(rewards_array),
                                            np.std(rewards_array)))

    print('Latex Print\n')
    print('$%.2f \\pm %.2f$ & %.2f \\\\\\hline' % (np.mean(rewards_array),
                                                   np.std(rewards_array),
                                                   np.sum(success_rate_array) * 100 / success_rate_array.size))

    # .....Statistics on Partitioned Episodes...
    print('Statistics on %d partitions' % n_partitions)

    r_partition_array = np.ones(n_partitions) * np.nan
    s_partition_array = np.ones(n_partitions) * np.nan
    l_partition_array = np.ones(n_partitions) * np.nan

    for i in range(n_partitions):
        r_partition_array[i] = np.mean(rewards_array[i * (n_episodes_per_partition): (i + 1) * n_episodes_per_partition])
        s_partition_array[i] = np.sum(success_rate_array[i * (n_episodes_per_partition): (i + 1) * n_episodes_per_partition])
        l_partition_array[i] = np.sum(landing_array[i * (n_episodes_per_partition): (i + 1) * n_episodes_per_partition])

    s_partition_array = s_partition_array*100/n_episodes_per_partition
    l_partition_array = l_partition_array*100/n_episodes_per_partition

    print('Partitioned Rewards = %.2f \\pm %.2f' % (np.mean(r_partition_array), np.std(r_partition_array)))
    print('Partitioned Success = %.2f \\pm %.2f' % (np.mean(s_partition_array), np.std(s_partition_array)))
    print('Partitioned Landing = %.2f \\pm %.2f' % (np.mean(l_partition_array), np.std(l_partition_array)))

    print('It took %.2f secs' % (t_f - t_0))

    print('Done')