import numpy as np
import copy


def mutate_pop(pop, params):
    """
    mutates pop and returns the mutated pop
    :param pop:
    :param params:
    :return: mutated_pop
    """
    parent_pop = copy.deepcopy(pop)
    pop_size = len(pop)
    child_pop = list(range(pop_size))

    for i in range(pop_size):
        child_pop[i] = mutate_ind(parent_pop[i], params)

    return child_pop


def mutate_ind(ind, params):
    mutated_ind = copy.deepcopy(ind)
    n_allowable_powers = len(params.allowable_powers)
    p_zero = params.p_zero
    [n_p, n_var] = ind.b_mat.shape

    for i in range(n_p):
        for j in range(n_var):
            r = np.random.rand()
            if r < params.p_mut:
                #..Do mutation....
                r_zero = np.random.rand()
                if r_zero < params.p_zero:
                    #..forcing an element to 0 value..
                    mutated_ind.b_mat[i,j] = 0
                else:
                    L_R = np.random.rand()
                    k_r = np.random.rand()
                    if k_r < params.beta_mut:
                        step_length = 1
                    else:
                        step_length = 2

                    if L_R < 0.5:#...step left..
                        step_length = step_length*(-1)

                    x_id = np.where(params.allowable_powers == ind.b_mat[i,j])[0][0]
                    mutated_id = x_id + step_length
                    if mutated_id < 0:
                        mutated_id = 0
                    elif mutated_id >= n_allowable_powers:
                        mutated_id = n_allowable_powers - 1

                    mutated_ind.b_mat[i,j] = params.allowable_powers[mutated_id]

    return mutated_ind

