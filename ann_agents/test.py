import gym
from ann_agents.PPO import PPO, Memory
from PIL import Image
import torch
from rl_envs.custom_envs import MyCartPole
import numpy as np
import os


def test():
    ############## Hyperparameters ##############
    random_states_flag = False#..we don't interact with the env if this is true...
    state_bounds_L = -1*np.ones(8)
    state_bounds_U = 1*np.ones(8)
    p_flag = 1#..store p-values....
    diff_tau = 0.0
    deterministic = True
    render = False
    save_data = True
    n_data_total = 1000
    n_data = 0
    n_episodes = 100
    max_timesteps = 1000
    balaced_data_flag = True
    #env_name = "MyCartPole"
    #env_name = "MountainCar-v0"
    #env_name = 'CartPole-v0'
    env_name = 'LunarLander-v2'
    #env_name = 'Acrobot-v1'
    extra_name = ''#'testing_'#'q_values_confident_00_' + 'random_'
    #save_dir = os.path.join('..', 'results','explain_ann_output')
    save_dir = os.path.join('..', 'datasets')

    # creating environment
    if env_name == 'MyCartPole':
        env = MyCartPole()#gym.make(env_name)
    else:
        env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    n_latent_var = 64           # number of variables in hidden layer
    lr = 0.0007
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    #############################################

    save_gif = False

    filename = "PPO_{}.pth".format(env_name)
    directory = ''#"./preTrained/"
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    
    ppo.policy_old.load_state_dict(torch.load(directory+filename))
    rewards_array = np.ones(n_episodes)*np.nan

    if balaced_data_flag == 1:
        n_pts_per_action = int(n_data_total/action_dim)
        action_counter = np.zeros(action_dim)
        n_data_total = int(n_pts_per_action*action_dim)
        performance_array = np.zeros((n_data_total, state_dim + 1))
    else:
        performance_array = np.zeros((n_data_total, state_dim + 1))

    if p_flag == 1:
        performance_array = np.zeros((n_data_total, state_dim + action_dim + 1))

    while True:
        if n_data >= n_data_total:
            break
        for ep in range(1, n_episodes+1):
            ep_reward = 0
            state = env.reset()
            for t in range(max_timesteps):
                if random_states_flag:
                    state = state_bounds_L + np.random.random(8)*(state_bounds_U - state_bounds_L)
                    state[-2] = np.random.randint(2)
                    state[-1] = np.random.randint(2)
                if deterministic:
                    action_array = ppo.policy_old.act_test(state, memory, p_flag=p_flag)
                    action = int(action_array[-1])
                else:
                    action = ppo.policy_old.act(state, memory)
                if balaced_data_flag == 1:
                    if action_counter[int(action)] < n_pts_per_action:
                        if p_flag == 1:
                            p_values = action_array[:-1]
                            sort_p_values = np.sort(p_values)
                            if sort_p_values[-1] - sort_p_values[-2] >= diff_tau:
                                performance_array[n_data, :] = np.concatenate([state, action_array])
                                action_counter[int(action)] += 1
                                n_data += 1  # ..update number of datapoints...
                        else:
                            performance_array[n_data, :] = np.concatenate([state, [action]])
                            action_counter[int(action)] += 1
                            n_data += 1  # ..update number of datapoints...
                else:
                    if p_flag == 1:
                        p_values = action_array[:-1]
                        sort_p_values = np.sort(p_values)
                        if sort_p_values[-1] - sort_p_values[-2] >= diff_tau:
                            performance_array[n_data, :] = np.concatenate([state, action_array])
                            n_data += 1  # ..update number of datapoints...
                    else:
                        performance_array[n_data,:] = np.concatenate([state, [action]])
                        n_data += 1  # ..update number of datapoints...

                print(f'n_data = {n_data}')

                state, reward, done, _ = env.step(action)
                ep_reward += reward
                if render:
                    env.render()
                if save_gif:
                     img = env.render(mode = 'rgb_array')
                     img = Image.fromarray(img)
                     img.save('./gif/{}.jpg'.format(t))
                if done:
                    break


                #print(f'n_data = {n_data}')
                if n_data >= n_data_total:
                    break

            if n_data >= n_data_total:
                print(f'n_data = {n_data}')
                break

            print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
            rewards_array[ep-1] = ep_reward
            print(f'n_data = {n_data}')
            ep_reward = 0
            env.close()

    # ...SAVING DATA....
    if save_data:
        if balaced_data_flag == 1:
            f_name = env_name + '_' + extra_name + str(n_data_total) + '_' + 'balanced_data' + '.data'
        else:
            f_name = env_name + '_' + extra_name + str(n_data_total) + '.data'
        print(f'Saving file with name: {f_name}')
        file_name = os.path.join(save_dir, f_name)
        np.savetxt(file_name, X=performance_array, delimiter=',')

    # ...Class Statistics..
    action_array = performance_array[:, -1]
    unique_actions = np.unique(action_array)
    class_distribution = -1 * np.ones(len(unique_actions))

    for i in range(len(unique_actions)):
        n_a = sum(action_array == unique_actions[i])
        class_distribution[i] = n_a

    print(f'Class Distribution: {class_distribution}')
    print('Rewards Stats')
    print('Min \t Max \t Avg \t Std')
    print('%.2f \t %.2f \t %.2f \t %.2f' % (np.min(rewards_array),
                                            np.max(rewards_array),
                                            np.mean(rewards_array),
                                            np.std(rewards_array)))

if __name__ == '__main__':
    test()
    
    
