"""
for asynchronous parallel evaluation of population individuals..
"""

import numpy as np

from iai_utils import iai_util_funcs as iai_funcs
from iai_ga.extract_vector import extract_vector
import multiprocessing as mp

pop_evaluated = []


def evaluate_ind(indiv, X_train, Y_train, params):
    # ..checks if an individual is already in the archive to avoid repeating computations...
    if indiv in params.archive:
        archive_id = params.archive.index(indiv)
        indiv = params.archive[archive_id]

    else:
        x_transformed = iai_funcs.transform_to_b_space(X_train, indiv.b_mat)
        #x_transformed = X_train[:,:-1]

        combined_array = np.concatenate((x_transformed,
                                         Y_train.reshape((len(Y_train), 1))),
                                        axis=1)

        w_all = iai_funcs.determine_weights_and_biases(x_transformed, Y_train, 0)

        w = w_all[:-1]
        bias = w_all[-1]
        indiv.w = w
        indiv.bias = bias

        indiv.fitness_ll = iai_funcs.compute_split_score(data=combined_array,
                                                          w=w,
                                                          bias=bias,
                                                          abs_flag=indiv.abs_flag)

        indiv.fitness_ul = compute_upper_level_fitness(indiv.b_mat, w)
        indiv.cons_vio_ul = indiv.fitness_ll - params.tau_impurity

    return indiv


def collect_indivs(indiv):
    global pop_evaluated
    pop_evaluated.append(indiv)


def evaluate_pop(pop, X_train, Y_train, params):
    """
    evaluates population members...
    each population individual is a an object
    :param params:
    :param pop:
    :param X_train:
    :param Y_train:
    :return:
    """
    global pop_evaluated
    pop_size = len(pop)

    #...initialzing parallel pool...
    pool = mp.Pool(mp.cpu_count())

    pop_evaluated = [pool.apply(evaluate_ind,
                                args=(indiv, X_train, Y_train, params))
                     for indiv in pop]

    """
    for i in range(pop_size):
        pool.apply_async(evaluate_ind,
                         args = (pop[i], X_train, Y_train, params),
                         callback=collect_indivs)
    """

    pool.close()
    #pool.join()

    pop = pop_evaluated
    if len(pop_evaluated) == 0:
        raise Exception('Evaluated Pop is Empty!')

    compute_net_fitness(pop)
    return pop


def compute_upper_level_fitness(b_mat, w):
    """
    :param b_mat:
    :param w:
    :return: number of active terms...
    """

    n_nonzero = 0
    n_prods = len(w)

    for i in range(n_prods):
        if w[i] != 0:
            n_nonzero += sum(b_mat[i,:] != 0)

    return n_nonzero


def compute_net_fitness(pop):
    """
    feasible_indivs.net_fitess = fitness
    infeasible_indivs.net_fitness = worst_feasible_fitness + cons_vio
    :param pop:
    :return: assignts net_fitess to the individuals...
    """

    pop_size = len(pop)
    fitness_vector = extract_vector(pop, 'fitness_ul')
    cons_vio_vector = extract_vector(pop, 'cons_vio_ul')

    if max(cons_vio_vector) <= 0:#..entire population is feasible!!!
        net_fitness = fitness_vector
    elif min(cons_vio_vector) > 0:#..entire population is infeasible!!!
        net_fitness = cons_vio_vector
    else:
        feasible_sol_fitness = fitness_vector[cons_vio_vector <= 0]
        worst_fitness = max(feasible_sol_fitness)

        net_fitness = fitness_vector
        net_fitness[cons_vio_vector > 0] = net_fitness[cons_vio_vector > 0] + worst_fitness

    for i in range(pop_size):
        pop[i].net_fitness = net_fitness[i]
