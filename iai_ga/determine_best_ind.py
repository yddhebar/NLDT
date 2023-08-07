import numpy as np
from iai_ga.extract_vector import extract_vector
from iai_ga.selection import select_best_of_two
from iai_ga.selection import select_best_of_two_modified
from iai_ga.selection import select_best_of_two_cons_vio_ad

def determine_best_ind(pop, best_ind, params = None):
    """

    :param pop:
    :param best_ind: current best-so-far individual
    :return: updated best_ind
    """
    if params.selection_method == 'normal':
        net_fitness_vec = extract_vector(pop,'net_fitness')
        curr_best_id = np.argmin(net_fitness_vec)
        curr_best = pop[curr_best_id]
        best_ind = select_best_of_two(curr_best, best_ind)
    elif params.selection_method == 'cons_ratio':
        for indiv in pop:
            best_ind = select_best_of_two_modified(best_ind, indiv)
    elif params.selection_method == 'cons_ratio_ad':
        for indiv in pop:
            best_ind = select_best_of_two_cons_vio_ad(best_ind, indiv)

    return best_ind

