import numpy as np
import sys
import os
sys.path.append(os.path.join('..','..'))
from rl_envs.custom_envs import Ford1DEnv
from rl_envs.custom_envs import MyCartPole
from rl_envs.custom_envs import PlanarSerialManipulator
import pickle
import matplotlib.pyplot as plt
import gym
from gym import wrappers
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib
import inspect


time_delay = 0.05
plot_flag = True
closed_loop_pruning = True
render_flag = True
tree_id = 0
fine_tuned_simulations = True
depth = 3
record_video = False
max_iters = 500
n_partitions = 50
n_episodes_per_partition = 100
n_episodes = n_partitions*n_episodes_per_partition
if record_video:
    n_episodes = 3
if plot_flag:
    n_episodes = 1
#env_name = 'MountainCar-v0'
#env_name = 'CartPole-v0'
#env_name = 'LunarLander-v2'
#env_name = 'Acrobot-v1'
n_links = 10
if n_links <= 5:
    torque_max = 200 * n_links  # ...for Planar Manipulator problem....
else:
    torque_max = 400 * n_links
env_name = 'PlanarManipulator' + '_' + str(n_links) + '_' + str(torque_max)
#env_name = "Ford1DEnv"

#data_name = 'CartPole-v0_10000'
#data_name = 'MountainCar' + '_balanced'# _q_values'#
#data_name = 'LunarLander-v2_q_values_confident_00_10000' + '_balanced_data'
#data_name = 'LunarLander-v2_10000' + '_balanced_data'
#data_name = 'Acrobot-v1_10000'# + '_balanced'
#data_name = 'PlanarManipulator_10000' + '_balanced_data'
#data_name = 'PlanarManipulator_' + str(n_links) + '_10000' + '_balanced_data'
data_name = 'PlanarManipulator_' + str(n_links) + '_' + str(torque_max) + '_10000' + '_balanced_data'
#data_name = 'Ford1DEnv_10000'
extra_name = '_classical' + '_dt_' + str(tree_id)
#extra_name = '_rga' + '_dt_' + str(tree_id)

action_array = np.ones(max_iters)*(-100)
time_array = np.arange(max_iters)
#extra_name = '_dt_' + str(tree_id)

if fine_tuned_simulations:
    tree_name = env_name + '_fine_tuned_from_env_depth_' + str(depth) + extra_name# + '_' + str(tree_id)
else:
    tree_name = data_name + extra_name + '_pruned'
    #tree_name = 'LunarLander_balanced_data_dt_0' + '_pruned'

if closed_loop_pruning:
    tree_name = tree_name + '_new_class'

# creating environment
if env_name == 'Ford1DEnv':
    env = Ford1DEnv(lead_profile='uniform')#, init_vel=0)
if 'PlanarManipulator' in env_name:
    env = PlanarSerialManipulator(render_flag=render_flag, n_links=n_links, torque_value=torque_max)
else:
    time_delay = 0.0
    env = gym.make(env_name)
    if record_video:
        #env = wrappers.Monitor(env, "./vid", video_callable=lambda episode_id: True, force=True)
        env = wrappers.Monitor(env, "./vid", force=True, video_callable=None, uid=env_name)

if render_flag or record_video:
    n_cpus = 1
else:
    n_cpus = np.min([50, mp.cpu_count()])

if n_cpus == 1:
    matplotlib.use('TkAgg')
    font = {'weight': 'bold',
            'size': 40}
    matplotlib.rc('font', **font)

rewards_array = np.ones(n_episodes)*np.nan
success_rate_array = np.zeros(n_episodes)
landing_array = np.zeros(n_episodes)

my_tree_loc = os.path.join('..','results', data_name ,tree_name + '.p')
my_agent = pickle.load(open(my_tree_loc, 'rb'))

print('Depth = %d' % my_agent.tree['max_depth'])
#time.sleep(6)

t_0 = time.time()

def run_simulation(e):
    print("Episode %d"% (e+1))
    s = env.reset()
    ep_r = 0
    success_ = 0
    landing_ = 0
    for i in range(max_iters):
        #print(f'iteration = {i}')
        if render_flag:
            if 'PlanarManipulator' not in env_name:
                env.render()
            if i == 0:
                print('Initial State = ' + str(s))
                #time.sleep(3)
            time.sleep(time_delay)
        action = int(my_agent.choose_action(s))
        action_array[i] = int(action)
        s_, r_, done, info = env.step(action)
        ep_r += r_
        if 'PlanarManipulator' in env_name:
            landing_ = env.num_steps
        if done:
            print('i = %d' % i)
            if 'Ford1D' in env_name:
                if i >= 799:
                    landing_array[e] = 1
                    success_rate_array[e] = 1

            if env_name == 'CartPole-v0':
                if i == 200-1:
                    print('Yes')
                    success_rate_array[e] = 1
                    success_ = 1
                print(info)

            if env_name == 'MountainCar-v0':
                if s_[0] >= 0.5 and s_[1] >= 0:
                    print('Yes')
                    success_rate_array[e] = 1
                    success_ = 1
                print(info)

            if env_name == 'Acrobot-v1':
                if i < 199:
                    success_ = 1

            if 'LunarLander' in env_name:
                if not env.env.lander.awake: #r_ == 100:
                    print('Yes')
                    success_rate_array[e]  = 1
                    success_ = 1
                else:
                    print('No')

                if (s_[-1] == 1 and s_[-2] == 1) or r_ == 100:
                    landing_array[e] = 1
                    landing_ = 1
                    print('But Yes')
                else:
                    #env.render()
                    print(s_)
                    print('Here')
            if 'PlanarManipulator' in env_name:
                if env.num_steps < max_iters:
                    success_ = 1
                print('Time Steps = %d' % env.num_steps)
            break
        s = s_

    rewards_array[e] = ep_r

    return np.array([ep_r, success_, landing_])


if __name__ == '__main__':
    if n_cpus > 1:
        with mp.Pool(processes=n_cpus) as pool:
            results = pool.map(run_simulation, list(range(n_episodes)))
    else:
        results = []
        for e in range(n_episodes):
            result = run_simulation(e)
            results.append(result)

    t_f = time.time()
    env.close()

    results = np.array(results)
    rewards_array = results[:,0]
    success_rate_array = results[:,1]
    landing_array = results[:,2]

    print('Success Rate = %d' % np.sum(success_rate_array))
    print('Landing Rate = %d' % np.sum(landing_array))

    #..number of time steps...
    print('\nNumber of time steps = min (%d), max (%d), %.2f +- %.2f\n' %
          (np.min(landing_array), np.max(landing_array), np.mean(landing_array), np.std(landing_array)))

    print('Rewards Stats')
    print('Min \t Max \t Avg \t Std')
    print('%.2f \t %.2f \t %.2f \t %.2f' % (np.min(rewards_array),
                                            np.max(rewards_array),
                                            np.mean(rewards_array),
                                            np.std(rewards_array)))

    print('Latex Print\n')
    print('$%.2f \\pm %.2f$ & %.2f \\\\\\hline' % (np.mean(rewards_array),
                                                    np.std(rewards_array),
                                                    np.sum(success_rate_array)*100/success_rate_array.size))

    #.....Statistics on Partitioned Episodes...
    if record_video is False:
        print('Statistics on %d partitions' % n_partitions)
        r_partition_array = np.ones(n_partitions)*np.nan
        s_partition_array = np.ones(n_partitions)*np.nan
        l_partition_array = np.ones(n_partitions)*np.nan

        for i in range(n_partitions):
            r_partition_array[i] = np.mean(
                rewards_array[i * (n_episodes_per_partition): (i + 1) * n_episodes_per_partition])
            s_partition_array[i] = np.sum(
                success_rate_array[i * (n_episodes_per_partition): (i + 1) * n_episodes_per_partition])
            l_partition_array[i] = np.sum(landing_array[i * (n_episodes_per_partition): (i + 1) * n_episodes_per_partition])

        s_partition_array = s_partition_array * 100 / n_episodes_per_partition
        l_partition_array = l_partition_array * 100 / n_episodes_per_partition

        print('Partitioned Rewards = %.2f \\pm %.2f' %(np.mean(r_partition_array), np.std(r_partition_array)))
        print('Partitioned Success = %.2f \\pm %.2f' % (np.mean(s_partition_array), np.std(s_partition_array)))
        print('Partitioned Landing = %.2f \\pm %.2f' % (np.mean(l_partition_array), np.std(l_partition_array)))

    print('It took %.2f secs' % (t_f - t_0))

    plt.plot(time_array[:env.num_steps], action_array[:env.num_steps], '-', linewidth=2)
    plt.xlabel('Time Steps')
    plt.ylabel('Action')
    plt.yticks(np.array([0,1,2]))
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95,
                        top=0.95, wspace=0.2, hspace=0.2)
    plt.grid()
    plt.show()

    print('Done')

