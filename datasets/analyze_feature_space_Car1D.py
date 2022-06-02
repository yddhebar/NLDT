import numpy as np
import pickle
import os
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


#..load data...
#data_file = os.path.join('CarFollowing_uniform_lead_acc_4vels_data_20000_balanced_random.data')
data_file = os.path.join('CarFollowing_uniform_lead_acc_4vels_data_20000_balanced_random.data')

my_data = pd.read_csv(data_file)


features = my_data.iloc[:,:-1]
action_array = my_data.iloc[:,-1]
print('Data Loaded')

unique_actions = pd.unique(action_array)
unique_actions = np.sort(unique_actions)
features = np.array(features)
#..Plotting...
font = {'size': 15}
matplotlib.rc('font', **font)
fig = plt.figure(1)
ax = plt.axes(projection='3d')

legend_list = []
n_actions = len(unique_actions)
for i in range(0, n_actions):
    c_a_array = action_array == unique_actions[i]
    x1 = features[c_a_array, 0]
    x2 = features[c_a_array, 1]
    x3 = features[c_a_array, 2]
    ax.scatter3D(x1, x2, x3, 'C' + str(i), alpha=(1/n_actions)*(n_actions - i))
    legend_list.append('Action ' + str(unique_actions[i]))


ax.legend(legend_list)


ax.set_xlabel('x0')
ax.set_ylabel('x1')
ax.set_zlabel('x2')


#..scatter plot...
plt.figure(2)

legend_list = []
for i in range(0,len(unique_actions)):
    c_a_array = action_array == unique_actions[i]
    x1 = features[c_a_array, 0]
    x2 = features[c_a_array, 1]
    plt.scatter(x1, x2, c ='C' + str(i), alpha=(1/n_actions)*(n_actions - i))
    legend_list.append('Action ' + str(unique_actions[i]))


plt.legend(legend_list)

plt.xlabel('x0')
plt.ylabel('x1')

plt.show()

print('End')
