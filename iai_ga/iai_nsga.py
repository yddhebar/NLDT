#..main file for Bilevel Algorithm...
import numpy as np
from iai_ga.iai_ga_classes import Parameters
from iai_ga.initialize_pop import initialize_pop
from iai_ga.repair_pop import repair_pop
from iai_ga.evaluate_pop import evaluate_pop# as evaluate_pop_serial
from iai_ga.evaluate_pop_parallel import evaluate_pop as evaluate_pop_parallel
from iai_ga.determine_best_ind import determine_best_ind
from iai_ga.evolve_pop import evolve_pop
from iai_ga.merge_and_reduce import merge_and_reduce_pop
from main_codes.parameter_file import GlobalParameters

global_params = GlobalParameters()

def bilevel_ga(X_train, Y_train, params = Parameters(), data_weights = None):
    """

    :param X_train:
    :param Y_train:
    :param params:
    :return:
    """
    params.n_var = X_train.shape[1]
    params.a_max = params.n_var
    params.abs_flag = 1
    params.p_mut = 1/params.n_var
    params.n_prod_terms = global_params.n_prod_terms
    params.selection_method = global_params.selection_method
    params.tau_impurity = global_params.tau_impurity
    params.pop_size = np.max([10, params.n_var]) #10*params.n_var#... this will slow down the algorithm miserably...
    all_gen_fitness = np.zeros(params.max_gen + 1)
    all_gen_cons_vio = np.zeros(params.max_gen + 1)

    print('gen = ',0)
    parent_pop = initialize_pop(params)
    parent_pop = repair_pop(parent_pop, params)
    parent_pop = evaluate_pop(parent_pop, X_train, Y_train, params, data_weights)
    best_so_far = parent_pop[0]
    best_so_far = determine_best_ind(parent_pop, best_so_far, params)
    all_gen_fitness[0] = best_so_far.fitness_ul
    all_gen_cons_vio[0] = best_so_far.cons_vio_ul

    print('best F_U: %d, best cons vio = %.2e'% (best_so_far.fitness_ul,
                                                 best_so_far.cons_vio_ul))

    if best_so_far.fitness_ul == 1 and best_so_far.cons_vio_ul <= 0:
        print('Bilevel GA converged!!')
        return best_so_far

    for gen in range(params.max_gen):

        print('gen = ', gen+1)

        child_pop = evolve_pop(parent_pop, params)
        child_pop = evaluate_pop(child_pop, X_train, Y_train, params, data_weights)
        parent_pop = merge_and_reduce_pop(parent_pop, child_pop)
        best_so_far = determine_best_ind(child_pop, best_so_far, params)
        all_gen_fitness[gen + 1] = best_so_far.fitness_ul
        all_gen_cons_vio[gen + 1] = best_so_far.cons_vio_ul

        print('best F_U: %d, best cons vio = %.2e' % (best_so_far.fitness_ul,
                                                      best_so_far.cons_vio_ul))

        if best_so_far.fitness_ul == 1 and best_so_far.cons_vio_ul <= 0:
            print('Bilevel GA converged!!')
            break

        if gen > 5:
            fitness_change = abs((all_gen_fitness[gen] - all_gen_fitness[gen-5])/all_gen_fitness[gen-5])
            cons_vio_change = abs((all_gen_cons_vio[gen] - all_gen_cons_vio[gen-5])/all_gen_cons_vio[gen-5])

            if fitness_change < 0.001 and (all_gen_cons_vio[gen - 5] <= 0):
                print('Bilevel GA converged!!')
                break

            if (fitness_change < 0.001) and (cons_vio_change < 0.001):
                print('Bilevel GA converged!!')
                break

    print('Done with Deriving Split Rule')

    #...Save Upperlevel optimization stats....
    print('Saving Generation Fitness and Cons. Vio.')

    data_array = np.vstack([np.arange(params.max_gen + 1), all_gen_fitness, all_gen_cons_vio]).transpose()
    np.savetxt('upper_level_fitness.txt', data_array, delimiter=', ')

    return best_so_far
