import numpy as np
from iai_ga.iai_ga_classes import Indiv

def initialize_pop(params):
    """
        %....INITIALIZE GA POPULATION of Block Matrices and abs_flag...
    """
    #val = np.random.random((n_samples, problem.n_var))
    n_vars = params.n_var
    n_prod_terms = params.n_prod_terms
    pop_size = params.pop_size
    allowable_powers = params.allowable_powers
    init_pop = []

    for i in range(pop_size):
        indiv = Indiv()
        b_mat = np.zeros([n_prod_terms, n_vars])
        indiv.abs_flag = 0
        indiv.b_mat = b_mat
        if params.abs_flag == 1:
            if np.random.rand() < 0.5:
                indiv.abs_flag = 1
        #..Add individual to the list of population
        init_pop.append(indiv)

    pop_id = 1
    i = 0

    for j in range(n_vars):
        for k in allowable_powers:
            if k == 0:
                continue

            init_pop[pop_id].b_mat[i,j] = k
            pop_id += 1
            break

        if pop_id >= pop_size:
            break

    return init_pop
