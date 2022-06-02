import numpy as np
from ann_agents import ann_agent_classes_funcs as agents
from rl_envs.custom_envs import CarFollowing
from rl_envs.custom_envs import MyCartPole
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

render_flag = False
p_values_flag = True#>...to indicate if we are storing p_values....
#agent_name = 'CartPole'
agent_name = 'CarFollowing'
env = CarFollowing(lead_profile='uniform')
#env = gym.make('CartPole-v0')
#env_to_wrap = MyCartPole()
#env = wrappers.Monitor(env_to_wrap, 'animations', force = True)
observation = env.reset()
max_iters = 1000
N_STATES = env.observation_space.shape[0]
N_ACTIONS = env.action_space.n
my_agent = agents.MyAgent(env = env, NET_TYPE= 'big')
weight_file_name = os.path.join(agent_name + '.pth')
my_agent.set_weights(weight_file_name)

state_array = np.zeros([max_iters, N_STATES])
if p_values_flag:
    action_array = np.ones([max_iters, N_ACTIONS + 1])
else:
    action_array = np.ones([max_iters, 1])
s = env.reset()
for i in range(max_iters):
    print(f'iteration = {i}')
    state_array[i, :] = s
    if render_flag:
        env.render()
    if p_values_flag:
        action_ = my_agent.get_action_array(s)
        action_array[i, :] = action_
        action = int(action_[-1])
    else:
        action = int(my_agent.choose_action(s))
        action_array[i - 1, :] = action

    s_, r_, done, info = env.step(action)

    if done:
        break

    s = s_

env.close()

#..Plotting..
time_array = np.arange(i)
state_array_trancated = state_array[:i,:]
#....Saving the episode data...
f_name = agent_name + '_' + 'episode.data'
file_name = os.path.join('..','datasets', f_name)
data_ = np.hstack([state_array_trancated, action_array[:i,:]])
np.savetxt(file_name,data_, delimiter=',')

plt.plot(time_array, state_array_trancated[:,0],'b-')
plt.xlabel('time steps')
plt.ylabel('Rel dist')
plt.title('Relative Distance Plot')
plt.grid()
plt.plot()
plt.show()

print('Done')

