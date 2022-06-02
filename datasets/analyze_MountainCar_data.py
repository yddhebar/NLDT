import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 15}

matplotlib.rc('font', **font)

trajectory_flag = False
balance_flag = False
data_name = 'MountainCar'
if trajectory_flag:
    data_name += '_trajectory'
if balance_flag:
    data_name += '_balanced'

my_data_full = np.loadtxt(data_name + '.data', delimiter=',')
if trajectory_flag:
    my_data = my_data_full[:,:-1]
    path_completion_array = my_data_full[:,-1]
else:
    my_data = my_data_full
n_features = 2
n_actions = 3

#...Plot trajectories...
def plot_trajectories(X):
    id_array = np.arange(path_completion_array.size)
    done_ids = id_array[path_completion_array == 1]
    n_episodes = done_ids.size
    id_plot = id_array[0:done_ids[0] + 1]
    plt.plot(X[id_plot,0], X[id_plot,1], 'k--', linewidth = 2)
    for e in range(n_episodes):
        if e < n_episodes - 1:
            id_plot = id_array[done_ids[e] + 1: done_ids[e+1] + 1]
            plt.plot(X[id_plot, 0], X[id_plot, 1], 'r--', linewidth=1)
        elif path_completion_array[-1] == 0:
            id_plot = id_array[done_ids[e] + 1:]
            plt.plot(X[id_plot, 0], X[id_plot, 1], 'r--', label='trajectories', linewidth=1)

    #...Completed Paths....
if trajectory_flag:
    print("Completed Paths = %d" % path_completion_array.sum())

X = my_data[:,:n_features]
Y = my_data[:,-1]
p_values = my_data[:,n_features:-1]

p_max_array = np.max(p_values,axis=1)
p_sort_mat = np.sort(p_values,axis = 1)

action_dict = {0:'left', 1:'do nothing', 2:'right'}
color_dict = {0:'blue', 1:'orange', 2:'green'}
plt.figure()
for i in range(3):
    plt.scatter(X[Y == i, 0], X[Y == i, 1], label = action_dict[i], alpha=1 - 0.2*i, c = color_dict[i], s = 30)#, edgecolors='black'
plt.xlabel('x-position')
plt.ylabel('x-velocity')
if trajectory_flag:
    plot_trajectories(X)
plt.legend()
plt.title('Action Plot')
plt.grid()

plt.figure()
plt.scatter(X[:,0], X[:,1], c = p_max_array)#, edgecolors='black')#, linewidths=1)#cmap = plt.cm.jet)
plt.colorbar()
plt.grid()
if trajectory_flag:
    plot_trajectories(X)
plt.xlabel('x-position')
plt.ylabel('x-velocity')
plt.title('Probability Value Plot')

p_diff_array = p_sort_mat[:,-1] - p_sort_mat[:,-2]
plt.figure()
plt.scatter(X[:,0], X[:,1], c = p_diff_array)#, edgecolors='black')#, linewidths=1)#cmap = plt.cm.jet)
plt.colorbar()
plt.grid()
if trajectory_flag:
    plot_trajectories(X)
plt.xlabel('x-position')
plt.ylabel('x-velocity')
plt.title('Probability Difference Value Plot')

#...plot trajectories...
if trajectory_flag:
    plt.figure()
    plot_trajectories(X)
    plt.title('Trajectories')
    plt.xlabel('x-position')
    plt.ylabel('x-velocity')
    plt.legend()
    plt.grid()


plt.show()