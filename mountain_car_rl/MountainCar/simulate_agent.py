import pickle
import gym
import numpy as np

n_episodes = 1000
render_flag = True

agent = pickle.load(open('mountain_car_agent.p','rb'))

env = gym.make('MountainCar-v0')
episode_rewards = np.zeros(n_episodes)

for i in range(n_episodes):
    print('Episode: %d' % (i+1))

    observation = env.reset()
    totalReward = 0

    while True:
        if render_flag:
            env.render()
        if totalReward == 0:
            # For step 1, only
            action = agent.start(observation)
            observation, reward, done, _ = env.step(action)
        else:
            action = agent.act(observation, reward, done)
            observation, reward, done, _ = env.step(action)

        totalReward += reward

        if done:
            print('Total Reward = %d' % totalReward)
            agent.end(reward)
            episode_rewards[i] = totalReward
            break


#..Print reward statistics..
print('Rewards Stats')
print('Min \t Max \t Avg \t Std')
print('%.2f \t %.2f \t %.2f \t %.2f' % (np.min(episode_rewards),
                                        np.max(episode_rewards),
                                        np.mean(episode_rewards),
                                        np.std(episode_rewards)))

env.close()