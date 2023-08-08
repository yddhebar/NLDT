#...Class objectes related to IAI_GA.....
import numpy as np

class Parameters:
    def __init__(self, abs_flag = 0,
                 a_max = 1,#..number of active terms per power law
                 n_prod_terms = 3,#number of power laws,
                 allowable_powers = [-3,-2,-1, 0, 1, 2, 3],#set of exponents
                 tau_impurity = 0.05,#..threshold on split-impurity for Upper Level Constraint..
                 p_x_over_ul = 0.9,#..upper level xover probability
                 p_zero = 0.75,#..used in mutation
                 p_mut = 0.03,#..mutation probability..
                 beta_mut = 0.75,#..this is used in mutation..
                 max_gen = 100,
                 pop_size = 10,
                 seed = np.random.randint(1000),
                 selection_method = None
                 ):

        self.n_var = 0
        self.abs_flag = abs_flag
        self.a_max = a_max
        self.n_prod_terms = n_prod_terms
        self.allowable_powers = np.array(allowable_powers)
        self.archive = []#..archive of individuals.. this updated for each node...
        self.p_x_over_ul = p_x_over_ul
        self.p_zero = p_zero
        self.p_mut = p_mut
        self.beta_mut = beta_mut
        self.max_gen = max_gen
        self.pop_size = pop_size
        self.seed = seed
        self.tau_impurity = tau_impurity
        self.selection_method = selection_method


class Indiv:
    #..population individual...
    def __init__(self):
        self.abs_flag = 0
        self.b_mat = None
        self.w = None
        self.bias = None
        self.fitness_ul = None
        self.fitness_ll = None
        self.cons_vio_ul = None
        self.cons_vio_ll = None
        self.net_fitness = None
        self.info = {}

    def __eq__(self, other):
        if not isinstance(other, Indiv):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.abs_flag == other.abs_flag and np.array_equal(self.b_mat, other.b_mat)
