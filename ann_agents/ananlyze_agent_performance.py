import numpy as np
from ann_agents import ann_agent_classes_funcs as agents
from rl_envs.custom_envs import CarFollowing
import os
import matplotlib.pyplot as plt

agent_name = 'CarFollowing_uniform_lead_acc_4vels'
env = CarFollowing(lead_profile='uniform', a_range=np.array([-2,-1,0,1]))
max_iters = 1000
n_episodes = 0
sliding_window_size = 100
rel_dist_array = np.ones(max_iters + 1)

my_agent = agents.MyAgent(env = env, NET_TYPE= 'big')
weight_file_name = os.path.join(agent_name + '.pth')
my_agent.set_weights(weight_file_name)


#...Trainig Performance...
total_reward_array = np.loadtxt('total_reward_array_' + agent_name + '.txt')
chasing_time_array = np.loadtxt('chasing_time_array_' + agent_name + '.txt')
total_episodes = len(total_reward_array)

mean_chasing_time_array = np.zeros(total_episodes)
std_low_array = np.zeros(total_episodes)
std_high_array = np.zeros(total_episodes)

episode_array = np.arange(len(total_reward_array))

for e in range(total_episodes):
    if e < sliding_window_size:
        mean_value = np.mean(chasing_time_array[:e])
        std_value = np.std(chasing_time_array[:e])
    else:
        mean_value = np.mean(chasing_time_array[e - sliding_window_size:e])
        std_value = np.std(chasing_time_array[e - sliding_window_size:e])

    mean_chasing_time_array[e] = mean_value
    std_low_array[e] = mean_value - std_value
    std_high_array[e] = mean_value + std_value

#..plotting...
plt.figure()
plt.plot(episode_array, mean_chasing_time_array,'r--')
plt.plot(episode_array,std_low_array,'b--')
plt.plot(episode_array, std_high_array, 'b--')
plt.xlabel('Episodes')
plt.ylabel('Mean Chasing Time (s)')
plt.title('Episodes Vs Mean Chasing Time')
#plt.show()


#...performance on one episode..
s = env.reset()
time_array = np.arange(max_iters + 1)*env.time_step
rel_dist_array[0] = s[0]
for i in range(max_iters):
    action = int(my_agent.choose_action(s))

    new_state, r_, done, info = env.step(action)
    s = new_state

    rel_dist_array[i + 1] = new_state[0]
    if done:
        print(f'Done at iteration {i}')
        break

#..plotting...
plt.figure()
plt.plot(time_array[:i], rel_dist_array[:i])
plt.xlabel('Time in (s)')
plt.ylabel('Rel Distance (m)')
plt.title('Times Vs Rel Distance')


#...time taken array....
time_taken_array = np.zeros(n_episodes)
episode_array = np.arange(n_episodes)
for e in range(n_episodes):
    print(f'Episode {e}')
    s = env.reset()
    for i in range(max_iters):
        action = int(my_agent.choose_action(s))

        new_state, r_, done, info = env.step(action)

        s = new_state

        if done:
            print(f'Done at iteration {i}')
            break
    time_taken_array[e] = env.time_taken

#..plotting...
plt.figure()
plt.plot(episode_array, time_taken_array)
plt.xlabel('Episode')
plt.ylabel('Time in (s)')
plt.title('Times taken for a Episode')

#..show allplots..
plt.show()
