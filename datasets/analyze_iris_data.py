import numpy as np
import pandas as pd
import pickle
import os
from iai_numpy.iai_utils.factory_utils import preprocess_class_labels
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


#..load data...
data_file = os.path.join('iris.data')

my_data = pd.read_csv(data_file, header = None)

features = my_data.iloc[:,:-1]
action_array = my_data.iloc[:,-1]
print('Data Loaded')

unique_actions = action_array.unique()

#..Plotting...
font = {'size': 15}
matplotlib.rc('font', **font)
fig = plt.figure(1)
ax = plt.axes(projection='3d')

legend_list = []
for i in range(len(unique_actions)):
    c_a_array = action_array == unique_actions[i]
    x1 = features[c_a_array][0]
    x2 = features[c_a_array][1]
    x3 = features[c_a_array][2]
    ax.scatter3D(x1, x2, x3, 'C' + str(i))
    legend_list.append(str(i) + '-' + str(unique_actions[i]))


ax.legend(legend_list)


ax.set_xlabel('x0')
ax.set_ylabel('x1')
ax.set_zlabel('x2')


#..scatter plot...
fig = plt.figure(2)
ax = plt.axes(projection='3d')

legend_list = []
for i in range(len(unique_actions)):
    c_a_array = action_array == unique_actions[i]
    x1 = features[c_a_array][1]
    x2 = features[c_a_array][2]
    x3 = features[c_a_array][3]
    ax.scatter3D(x1, x2, x3, 'C' + str(i))
    legend_list.append(str(i) + '-' + str(unique_actions[i]))


ax.legend(legend_list)


ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')

plt.show()

print('End')
