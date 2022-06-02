import numpy as np

class UpperLevelParameters():
    #...Upper Level GA parameters..
    def __init__(self, n_features = 1,
                 beta = 3,
                 p_xover = 0.9):
        self.n_features = n_features
        self.n_prods = 3
        self.abs = 0
        self.exponents = np.arange(-3,4,1)#..Allowable set of exponents...
        self.pop_size = 30
        self.max_gen = 50
        self.max_active_terms_per_product_rule = n_features
        self.B_mat = np.zeros([self.n_prods, self.n_features])
        self.beta = beta
        self.p_xover = p_xover


class PIDParameters():
    EPS = np.finfo(float).eps
    #..PID Parameters...
    def __init__(self,
                 n_features = 1,
                 ):
        self.n_states = n_features
        self.n_vars = 3*self.n_states + 2
        self.u_bound = np.array((self.n_vars)*[1])
        self.l_bound = np.array((self.n_vars)*[-1])
        self.l_bound[-1] = self.EPS

