import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.special import softmax


class DQN(object):
    def __init__(self,
                 env={},
                 BATCH_SIZE=32,
                 LR = 0.01,  # ....learning rate
                 EPSILON=0.9,  # ...epsilon greedy policy
                 GAMMA=0.9,  # ...discount factor
                 TARGET_REPLACE_ITER = 100,  # .... Target-net update frequency...
                 MEMORY_CAPACITY=2000,
                 NET_TYPE = None):  #....Replay Memory Capacity...):

        self.env = env
        self.BATCH_SIZE = BATCH_SIZE
        self.LR = LR
        self.EPSILON = EPSILON
        self.GAMMA = GAMMA
        self.TARGET_REPLACE_ITER = TARGET_REPLACE_ITER
        self.MEMORY_CAPACITY = MEMORY_CAPACITY

        #...Extracting environment related variables...
        self.N_ACTIONS = env.action_space.n
        self.N_STATES = env.observation_space.shape[0]
        self.ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) \
            else env.action_space.sample().shape  # ...to confirm the shape..

        self.NET_TYPE = NET_TYPE
        if NET_TYPE == 'small':
            self.eval_net, self.target_net = Net_small(env=env), Net_small(env=env)
        elif NET_TYPE == 'big':
            self.eval_net, self.target_net = Net_big(env=env), Net_big(env=env)

        self.learn_step_counter = 0 #...for target updating..
        self.memory_counter = 0 #.. for storing memory...
        self.memory = np.zeros((MEMORY_CAPACITY, self.N_STATES*2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self,x):
        x = torch.unsqueeze(torch.FloatTensor(x),0)

        if np.random.uniform() < self.EPSILON: #..greedy policy...
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value,1)[1].data.numpy()#..simplify...
            action = action[0] if self.ENV_A_SHAPE == 0 else action.reshape(self.ENV_A_SHAPE)#..argmax index..
        else: #..random action..
            action = np.random.randint(0,self.N_ACTIONS)
            action = action if self.ENV_A_SHAPE == 0 else action.reshape(self.ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s,[a,r],s_))

        #..replace old memory with new memory...
        #...its a cyclic memory... nice!!!!
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index,:] = transition
        self.memory_counter += 1

    def learn(self):
        #..target parameter update...
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        #... sample batch of unique transitions..
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE, replace=False)
        b_memory = self.memory[sample_index,:]
        b_s = torch.FloatTensor(b_memory[:,:self.N_STATES])#..states#.....
        b_a = torch.LongTensor(b_memory[:,self.N_STATES:self.N_STATES + 1].astype(int))#..actions in int
        b_r = torch.FloatTensor(b_memory[:,self.N_STATES + 1:self.N_STATES+2])#...rewards..#...
        b_s_ = torch.FloatTensor(b_memory[:,-self.N_STATES:])#..next state...#...

        #.. q_eval w.r.t. action in experience..
        q_eval = self.eval_net(b_s).gather(1,b_a) #..this is of shape (batch,1)#..breakpoint...
        q_next = self.target_net(b_s_).detach()#...detaching from graph to avoid backpropagation..
        q_target = b_r + self.GAMMA*q_next.max(1)[0].view(self.BATCH_SIZE,1) #..shape of (batch,1)#..simplify..
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model_parameters(self,filename):
        torch.save(self.eval_net.state_dict(), filename)


class MyAgent:
    def __init__(self, eps=0.9, env = {}, NET_TYPE = None):
        self.EPSILON = eps  # ...epsilon greedy policy
        self.env = env
        self.N_ACTIONS = env.action_space.n
        self.N_STATES = env.observation_space.shape[0]
        self.ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) \
            else env.action_space.sample().shape  # ...to confirm the shape..

        if NET_TYPE == 'small':
            self.my_net = Net_small(env=env)
        elif NET_TYPE == 'big':
            self.my_net = Net_big(env=env)

    def set_weights(self, filename=''):
        self.my_net.load_state_dict(torch.load(filename))

    def choose_action(self,x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)

        if np.random.uniform() < self.EPSILON:  # ..greedy policy...
            actions_value = self.my_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()  # ..simplify...
            action = action[0] if self.ENV_A_SHAPE == 0 else action.reshape(self.ENV_A_SHAPE)  # ..argmax index..
        else:  # ..random action..
            action = np.random.randint(0, self.N_ACTIONS)
            action = action if self.ENV_A_SHAPE == 0 else action.reshape(self.ENV_A_SHAPE)
        return action

    def get_action_array(self,x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)

        actions_value = self.my_net.forward(x)
        action = torch.max(actions_value, 1)[1].data.numpy()  # ..simplify...
        action = action[0] if self.ENV_A_SHAPE == 0 else action.reshape(self.ENV_A_SHAPE)  # ..argmax index..

        action_value_numpy = softmax(actions_value.detach().numpy())
        return np.concatenate([action_value_numpy[0], [action]])


class Net_small(nn.Module):
    def __init__(self, env={}):
        super(Net_small,self).__init__()
        N_ACTIONS = env.action_space.n
        N_STATES = env.observation_space.shape[0]
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0,0.1) #..normal initialization...
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1) #..normal initialization...

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class Net_big(nn.Module):
    def __init__(self, env={}):
        super(Net_big,self).__init__()
        N_ACTIONS = env.action_space.n
        N_STATES = env.observation_space.shape[0]
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0,0.1) #..normal initialization...
        self.fc2 = nn.Linear(50, 50)
        self.fc2.weight.data.normal_(0, 0.1)  # ..normal initialization...
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1) #..normal initialization...

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


def compute_total_reward(ann_agent, env, max_iter = 1000, render_flag = False):
    """

    :param tree: Trained RL IAI Tree...
    :param max_iter: maximum number of iterations before episode terminates
    :return: total reward collected
    """
    s = env.reset()
    total_reward = 0
    for i in range(max_iter):
        if render_flag:
            env.render()
        action = int(ann_agent.choose_action(s))

        new_state, r_, done, info = env.step(action)

        total_reward += r_

        if done:
            break

        s = new_state

    return total_reward
