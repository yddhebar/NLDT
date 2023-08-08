import numpy as np
import copy
import random


def cross_over_pop(pop, params):
    """
    conducts crossover on the population and returns the child pop
    :param pop:
    :param params:
    :return: child_pop
    """
    parent_pop = copy.deepcopy(pop)
    pop_size = len(parent_pop)
    child_pop = list([0]*pop_size)
    a1 = list(range(pop_size))
    random.shuffle(a1)

    if pop_size%2 == 0:
        i_o = 0
    else:
        i_o = 1
        child_pop[pop_size-1] = parent_pop[pop_size-1]

    for i in range(0, pop_size - i_o, 2):
        [child_pop[i], child_pop[i+1]] = crossover(parent_pop[a1[i]], parent_pop[a1[i+1]], params)

    return child_pop


def crossover(parent1, parent2, params):
    """
    conducts crossover btw parent1 and parent2
    :param params:
    :param parent1:
    :param parent2:
    :return: [child1, child2]
    """
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    w1 = abs(parent1.w)
    w2 = abs(parent2.w)

    s_w1_ids = np.argsort(w1)
    s_w2_ids = np.argsort(w2)

    r = np.random.rand()

    if r < params.p_x_over_ul:
        p1_mat = parent1.b_mat
        p2_mat = parent2.b_mat
        [n_p, n_var] = p1_mat.shape

        temp1_mat = np.zeros([n_p, n_var])
        temp2_mat = np.zeros([n_p, n_var])

        #..for loop for termwise crossover..
        for i in range(n_p):
            for j in range(n_var):
                r2 = np.random.rand()
                if r2 < 0.5:
                    temp1_mat[i,j] = p1_mat[s_w1_ids[i],j]
                    temp2_mat[i,j] = p2_mat[s_w2_ids[i],j]
                else:
                    temp1_mat[i, j] = p2_mat[s_w2_ids[i], j]
                    temp2_mat[i, j] = p1_mat[s_w1_ids[i], j]

        child1.b_mat = temp1_mat
        child2.b_mat = temp2_mat

    return [child1, child2]