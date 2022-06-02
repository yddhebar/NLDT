import numpy as np
import pickle
import os
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


#..load data...
data_file = os.path.join('CartPole_500_Data_10000.p')
my_data = pickle.load(open(data_file, 'rb'))


features = my_data[:,:-1]
action_array = my_data[:,-1]
print('Data Loaded')


#..Plotting...
font = {'size': 15}
matplotlib.rc('font', **font)
fig = plt.figure(1)
ax = plt.axes(projection='3d')

c_a_array = action_array == 0
x1 = features[c_a_array,0]
x2 = features[c_a_array,1]
x3 = features[c_a_array,2]
ax.scatter3D(x1, x2, x3,'ro')
x1 = features[~c_a_array,0]
x2 = features[~c_a_array,1]
x3 = features[~c_a_array,2]
ax.scatter3D(x1, x2, x3,'bo')
ax.set_xlabel('x0')
ax.set_ylabel('x1')
ax.set_zlabel('x2')

#..Plotting...
fig = plt.figure(2)
ax = plt.axes(projection='3d')

c_a_array = action_array == 0
x2 = features[c_a_array,1]
x3 = features[c_a_array,2]
x4 = features[c_a_array,3]
ax.scatter3D(x2,x3, x4,'ro')
x2 = features[~c_a_array,1]
x3 = features[~c_a_array,2]
x4 = features[~c_a_array,3]
ax.scatter3D(x2, x3, x4,'ko')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')

plt.show()

print('End')
