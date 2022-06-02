import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

rel_dist_array = np.loadtxt('rel_dist_array.txt', delimiter=',')

print('Printing...')
rel_dist_array = rel_dist_array[rel_dist_array >= 0]

time_array = np.array(range(len(rel_dist_array)))

plt.plot(time_array, rel_dist_array, 'b*')
plt.show()

print('End')