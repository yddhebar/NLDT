import numpy as np
import time
from iai_utils import iai_util_funcs as iai_funcs
from iai_ga.extract_vector import extract_vector


def evaluate_pop(pop, X_train, Y_train, params, data_weights = None):
    """
    evaluates population members...
    each population individual is a an object
    :param params:
    :param pop:
    :param X_train:
    :param Y_train:
    :return:
    """
    pop_size = len(pop)
    t_0 = time.perf_counter()
    for i in range(pop_size):
        #..checks if an individual is already in the archive to avoid repeating computations...
        if pop[i] in params.archive:
            archive_id = params.archive.index(pop[i])
            pop[i] = params.archive[archive_id]

        else:
            x_transformed = iai_funcs.transform_to_b_space(X_train, pop[i].b_mat)

            if data_weights is None:
                combined_array = np.concatenate((x_transformed,
                                                 Y_train.reshape((len(Y_train),1))),
                                                axis = 1)
            else:
                combined_array = np.concatenate((x_transformed,
                                                 data_weights.reshape((len(Y_train), 1)),
                                                 Y_train.reshape((len(Y_train), 1))),
                                                axis=1)

            w_all = iai_funcs.determine_weights_and_biases(x_transformed, Y_train, pop[i].abs_flag, data_weights)
            #w_all = np.random.random(params.n_prod_terms + 1)
            #w_all = res[0]
            #f_ll_best = res[1]

            if pop[i].abs_flag == 0:
                w = w_all[:-1]
                bias = w_all[-1]
            elif pop[i].abs_flag == 1:
                w = w_all[:-2]
                bias = w_all[-2:]

            pop[i].w = w
            pop[i].bias = bias

            #pop[i].fitness_ll = f_ll_best


            pop[i].fitness_ll = iai_funcs.compute_split_score(data=combined_array,
                                                              w=w,
                                                              bias=bias,
                                                              abs_flag=pop[i].abs_flag)


            n_active = compute_upper_level_fitness(pop[i].b_mat, w)
            pop[i].fitness_ul = compute_upper_level_fitness(pop[i].b_mat, w)
            #pop[i].fitness_ul = (1 + pop[i].fitness_ll)*(n_active**0.25)
            pop[i].cons_vio_ul = pop[i].fitness_ll - params.tau_impurity

    t_f = time.perf_counter()
    print('Time Taken = %.2fs'% (t_f - t_0))
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
        net_fitness[cons_vio_vector > 0] = cons_vio_vector[cons_vio_vector > 0] + worst_fitness

    for i in range(pop_size):
        pop[i].net_fitness = net_fitness[i]
