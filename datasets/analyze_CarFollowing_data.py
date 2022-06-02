import numpy as np
import matplotlib
import matplotlib.pyplot as plt
font = {'family' : 'normal',
        'size'   : 26}
matplotlib.use('Qt5Agg')
matplotlib.rc('font', **font)

data_name = 'CarFollowing_10000'

data = np.loadtxt(data_name + '.data', delimiter=',')

X = data[:,:3]
Y = data[:,-1]
n_pts= Y.size

plt.figure()
plt.plot(X[:,0], np.ones(n_pts), '*')
plt.xlabel('Distance')
plt.title('Distance')
plt.grid()
plt.hist(X[:,0], bins= 100)

plt.figure()
plt.plot(X[:,1], np.ones(n_pts), '*')
plt.xlabel('Velocity')
plt.grid()
plt.title('Velocity')

plt.hist(X[:,1], bins= 100, color='blue')

plt.show()
print('End')