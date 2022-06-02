import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 30}

matplotlib.rc('font', **font)

#task = 'CartPole'
task = 'CarFollowing'

time_step = 0.01
d_safe = 30
d_max = 150

if task == 'CartPole':
    x = np.arange(-d_safe, d_safe, time_step)
    y = np.exp(-((abs(x))/d_safe)**0.5)

elif task == 'CarFollowing':
    x = np.arange(0,d_max, time_step)
    #y = 1*np.exp(-(x**0.25))
    y = d_max - abs(x - d_safe) + (d_max/2)*np.exp(-(((abs(x - d_safe))/d_safe)**1))
    y[x < d_safe] = 1.5*d_max*(x [x < d_safe])/d_safe

#..plotting...
plt.plot(x, y, linewidth=2, label = 'Reward Value')
if 'CarFollowing' in task:
    plt.plot([d_safe, d_safe], [np.min(y), np.max(y)], 'r--', linewidth=2)
    plt.text(x = d_safe + 2, y = 0, s = r'$d_{safe}$')
    plt.legend()
plt.xlabel(r'Relative Distance')
plt.ylabel('Reward')
plt.grid()
plt.title('Reward Function for CarFollowing-S Environment')
plt.show()


print('End')