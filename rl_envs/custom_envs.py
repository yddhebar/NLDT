import gym
from gym import spaces
from gym.envs.classic_control.cartpole import CartPoleEnv
from gym.envs.classic_control.mountain_car import MountainCarEnv
from gym.utils import seeding
import numpy as np


class MyCartPole(CartPoleEnv):
    def __init__(self):
        super().__init__()
        self.num_steps = 0
        self.time_taken = 0

    def reset(self):
        s = super().reset()
        self.num_steps = 0
        self.time_taken = 0
        return s

    def step(self, action):
        s_, r, done, info = super().step(action=action)

        self.num_steps += 1

        # ..modify the reward...
        x, x_dot, theta, theta_dot = s_
        #r1 = (self.x_threshold - abs(x)) / self.x_threshold - 0.8
        #r2 = (self.theta_threshold_radians - abs(theta)) / self.theta_threshold_radians - 0.5

        #y = np.exp(-((abs(x))/d_safe)**0.25) - 0.5
        r1 = np.exp(-(abs(x)/self.x_threshold)**0.5)
        r2 = np.exp(-(abs(theta) / self.theta_threshold_radians) ** 0.5)
        r = r1 + r2

        if done:
            self.time_taken = self.num_steps * self.tau

        return s_, r, done, info


class MyMountainCar(MountainCarEnv):
    def __init__(self):
        super().__init__()
        self.num_steps = 0
        self.time_taken = 0

    def reset(self):
        s = super().reset()
        self.num_steps = 0
        self.time_taken = 0
        return s

    def step(self, action):
        s_, r, done, info = super().step(action=action)

        self.num_steps += 1

        # Adjust reward based on car position
        reward = s_[0] + 0.5 #- (action-1)**2

        # Adjust reward for task completion
        if s_[0] >= 0.5:
            reward += 1
        r = reward
        #r = s_[0] + 0.5*r

        if self.num_steps >= 200:
            done = True

        if done:
            self.time_taken = self.num_steps

        return s_, r, done, info


class CarFollowing(CartPoleEnv):
    metadata = {'render.modes':['human']}

    def __init__(self,
                 factor = 1,
                 a_range = np.array([-1,1]),
                 v_range = np.array([0,32]),
                 d_safe = 30,
                 d_out_of_range = 150,
                 lead_profile = None,#..should be uniform or discrete
                 time_step = 0.25,
                 t_max = 200,
                 d_collision = 0
                 ):
        super(CarFollowing, self).__init__()

        if lead_profile is None:
            raise Exception('Supply the lead_profile')

        #..Important Environment Parameters...
        self.factor = factor
        self.a_range = a_range*factor#...possible values of acceleration of ego car
        self.pv_a_range = a_range*factor#...possible values of acceleration of pv car
        self.v_range = v_range#...possible  values of velocity..
        self.d_range = np.array([0, np.finfo(np.float32).max])#..possible values of distance..
        self.d_safe = d_safe#...safe distance....
        self.d_collision_dist = d_collision#... if rel_dist <= 0 than stop
        self.d_out_of_range = d_out_of_range#...if rel_dist > d_out_of_range... stop...
        self.Th = 1.5#...time headway...
        self.time_step = time_step#...seconds between state updates...
        self.t_max = t_max  # ..time in secs...
        self.lead_profile = lead_profile#...should be either 'uniform' or descrete
        # ..define action and observation space
        # ..they must be gym.spaces objects....
        #...example when using discrete actions..
        self.action_space = spaces.Discrete(len(self.a_range))

        #..obs_space = d, v, a_prev...
        lb = np.array([self.d_range[0], self.v_range[0],self.a_range[0]])
        ub = np.array([self.d_range[1], self.v_range[1],self.a_range[-1]])

        self.observation_space = spaces.Box(low = lb,
                                            high= ub,
                                            dtype=np.float32)

        #..Ego car and PV car settings..

        self.pv_car = None
        self.ego_car = None
        self.state = None

        #...Counter...
        self.total_steps = 0
        self.time_taken = 0

    def reset(self):
        #...reset the environment...

        pv_vel = 0

        if self.lead_profile == 'discrete':
            pv_acc = np.random.choice(self.a_range)#(self.a_range[0], self.a_range[1])
        elif self.lead_profile == 'uniform':
            pv_acc = np.random.uniform(self.pv_a_range[0], self.pv_a_range[-1])

        factor = 0.01
        pv_d = np.random.uniform(self.d_safe - factor*self.d_safe, self.d_safe + factor*self.d_safe)
        self.pv_car = np.array([pv_d, pv_vel, pv_acc])
        self.ego_car = np.array([0, 0, np.random.choice(self.a_range)])

        self.state =  self.pv_car - self.ego_car
        self.state[2] = self.ego_car[2]
        self.total_steps = 0
        return self.state

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self.total_steps += 1

        #....Ego Car dynamics....
        d_ego_prev, v_ego_prev, a_ego_prev = self.ego_car
        a_ego_next = self.a_range[action]
        d_ego_next = d_ego_prev + v_ego_prev*self.time_step
        v_ego_next = v_ego_prev + a_ego_next*self.time_step

        if v_ego_next > self.v_range[1]:
            v_ego_next = self.v_range[1]
        if v_ego_next < self.v_range[0]:
            v_ego_next = self.v_range[0]


        #..PV Car dynamics..
        d_pv_prev, v_pv_prev, a_pv_prev = self.pv_car

        if self.lead_profile == 'discrete':
            a_pv_next = np.random.choice(self.pv_a_range)
        elif self.lead_profile == 'uniform':
            a_pv_next = np.random.uniform(self.pv_a_range[0], self.pv_a_range[-1])
        d_pv_next = d_pv_prev + v_pv_prev*self.time_step
        v_pv_next = v_pv_prev + a_pv_next*self.time_step

        if v_pv_next > self.v_range[1]:
            v_pv_next = self.v_range[1]
        if v_pv_next < self.v_range[0]:
            v_pv_next = self.v_range[0]

        self.ego_car = np.array([d_ego_next, v_ego_next, a_ego_next])
        self.pv_car = np.array([d_pv_next, v_pv_next, a_pv_next])

        #...computing state...
        rel_dist = d_pv_next - d_ego_next
        rel_vel = v_pv_next - v_ego_next

        self.state = np.array([rel_dist, rel_vel, a_ego_next])

        self.time_taken = self.total_steps * self.time_step
        done = False
        if rel_dist <= self.d_collision_dist:
            #print('Car Collision!!!!')
            #print(f'Time taken is {self.time_taken} secs')
            done = True

        if rel_dist > self.d_out_of_range:
            #print('Car out of Range!!!!')
            #print(f'Time taken is {self.time_taken} secs')
            done = True


        if self.time_taken >= self.t_max:
            #print(f'The car successfully chased for {self.time_taken} secs')
            done = True

        #reward = (self.d_out_of_range**2 - (rel_dist - self.d_safe)**2)/(self.d_out_of_range**2)#  1/(1 + abs(rel_dist - self.d_safe)**0.33)
        #reward = np.exp(-(((rel_dist - self.Th*self.d_safe)**2))/(2*self.Th*self.d_safe)) - 1
        #reward = np.exp(-((abs(rel_dist - self.Th * self.d_safe) ** 0.5)) / (2 * self.Th * self.d_safe))#..reward 3..
        #reward = np.exp(-((abs(rel_dist - self.Th * self.d_safe) ** 1)) / (2 * self.Th * self.d_safe))
        #reward = np.exp(-(((abs(rel_dist -  self.d_safe))) / (self.d_safe)) ** 0.5)#...reward 5
        #reward = np.exp(-(((abs(rel_dist - self.d_safe))) / (self.d_safe)) ** 0.25) - 0.5  # ...reward 6
        #reward = np.exp(-(((abs(rel_dist - self.d_safe))) / (self.d_safe)) ** 2)  # ...reward 7

        #..reward 8...
        reward = self.d_out_of_range - abs(rel_dist - self.d_safe) + (self.d_out_of_range/2)*(np.exp(-(abs(rel_dist - self.d_safe)/self.d_safe)))
        if rel_dist < self.d_safe:
            reward = 1.5*self.d_out_of_range*(rel_dist)/(self.d_safe)

        #...returns state, reward, done, info.....
        return self.state, reward, done, {}

