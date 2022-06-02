import gym
from ann_agents.PPO_continuous import PPO, Memory
from PIL import Image
import torch
import numpy as np
from rl_envs.custom_envs import CarFollowing_Continuos

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test():
    ############## Hyperparameters ##############
    env_name = "CarFollowingContinuous"
    #env = gym.make(env_name)
    env = CarFollowing_Continuos()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    n_episodes = 1  # num of episodes to run
    max_timesteps = 1000  # max timesteps in one episode
    render = True  # render the environment
    save_gif = False  # png images are saved in gif folder

    # filename and directory to load model from
    filename = "PPO_continuous_" + env_name + ".pth"
    directory = ''

    action_std = 0.5  # constant std for action distribution (Multivariate Normal)
    K_epochs = 80  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr = 0.0003  # parameters for Adam optimizer
    betas = (0.9, 0.999)
    #############################################

    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    ppo.policy_old.load_state_dict(torch.load(directory + filename))

    rel_dist_array = -1*np.ones(max_timesteps)

    for ep in range(1, n_episodes + 1):
        ep_reward = 0
        state = env.reset()
        for t in range(max_timesteps):
            rel_dist_array[t] = state[0]
            action = ppo.select_action(state, memory)
            state, reward, done, _ = env.step(action)
            ep_reward += reward

            if done:
                break

        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        ep_reward = 0
        np.savetxt('rel_dist_array.txt', rel_dist_array, delimiter=',')




if __name__ == '__main__':
    test()

