import numpy as np
from matplotlib import pyplot as plt

# Plot the number of steps per episode
episode_rewards = np.load("rewardPerEpisode.npy")
lmda = 0.90
plt.plot(np.arange(len(episode_rewards)), episode_rewards, 
                    label='Sarsa Lambda\nNum Episodes= {}\nLast 100 Avg Reward= {}'.format(len(episode_rewards), round(np.average(episode_rewards[len(episode_rewards)-99:]), 2 ) ) )
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()

# print(np.average(episode_rewards[len(episode_rewards)-100:]))

plt.show()