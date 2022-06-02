import numpy as np
import gym
import os
import sys
sys.path.append(os.path.join('..'))
from rl_envs.custom_envs import CarFollowing
from rl_envs.custom_envs import MyCartPole
import ann_agents.ann_agent_classes_funcs as agents
import multiprocessing as mp
import time

#...Initialize Variables...
env = CarFollowing(lead_profile='uniform', a_range= np.array([-1,1]))
agent_name = 'CarFollowing'
extra_string = ''
my_agent = agents.MyAgent(env = env, NET_TYPE= 'big')
weight_file_name = os.path.join(agent_name + extra_string + '.pth')
my_agent.set_weights(weight_file_name)

#n_episodes = 1000
n_partitions = 50
n_episodes_per_partition = 100
n_episodes = n_partitions*n_episodes_per_partition
max_iter = 800
rewards_array = np.zeros(n_episodes)#..collectes total rewards earned during each episode...
render_flag = False


def run_simulation(e):
#for e in range(n_episodes):
    print(f'Episode {e}')
    ep_reward = agents.compute_total_reward(ann_agent= my_agent,
                                                 max_iter= max_iter,
                                                 env= env,
                                                 render_flag=render_flag)
    return ep_reward


if __name__ == '__main__':
    n_cpus = np.min([50, mp.cpu_count()])
    with mp.Pool(processes=n_cpus) as pool:
        results = pool.map(run_simulation, list(range(n_episodes)))

    t_f = time.time()
    env.close()

    rewards_array = np.array(results)
    #rewards_array = results[:,0]

    print('Max Score = %.2f' % np.max(rewards_array))
    print('Mean Score = %.2f' % np.mean(rewards_array), '\\pm %.2f' % np.std(rewards_array))
    print('Min Score = %.2f' % np.min(rewards_array))
    print('Avg +- STD = $%.2f \\pm %.2f$' % (np.mean(rewards_array),
                                             np.std(rewards_array)))

    # .....Statistics on Partitioned Episodes...
    print('Statistics on %d partitions' % n_partitions)
    r_partition_array = np.ones(n_partitions) * np.nan

    for i in range(n_partitions):
        r_partition_array[i] = np.mean(
            rewards_array[i * (n_episodes_per_partition): (i + 1) * n_episodes_per_partition])

    print('Partitioned Rewards = %.2f \\pm %.2f' % (np.mean(r_partition_array), np.std(r_partition_array)))




