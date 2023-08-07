import numpy as np
import random
import copy

from iai_ga.evaluate_pop import compute_net_fitness


def selection(pop_in, params):
    """
    does binary tournament selection on population
    :param pop:
    :return: selected list of individuals...
    """
    pop = copy.deepcopy(pop_in)
    pop_size = len(pop)
    a1 = list(range(pop_size))
    a2 = list(range(pop_size))
    random.shuffle(a1)
    random.shuffle(a2)
    sel_pop = list([0]*pop_size)

    for i in range(pop_size):
        #...encouraging "normal" selection to facilitate exploration..
        sel_pop[i] = select_best_of_two(pop[a1[i]], pop[a2[i]])
        '''
        if params.selection_method == 'normal':
            sel_pop[i] = select_best_of_two(pop[a1[i]], pop[a2[i]])
        elif params.selection_method == 'cons_ratio':
            sel_pop[i] = select_best_of_two_modified(pop[a1[i]], pop[a2[i]])
        '''

    return sel_pop


def select_best_of_two_modified(ind1, ind2):
    ind_list = [ind1, ind2]
    cons_vio_vec = np.array([ind1.cons_vio_ul, ind2.cons_vio_ul])
    obj_value_vec = np.array([ind1.fitness_ul, ind2.fitness_ul])

    if np.min(cons_vio_vec) <= 0:
        if np.max(cons_vio_vec) <= 0:#..both are feasible...
            return ind_list[np.argmin(obj_value_vec)]

    cons_vio_ratio = 100*np.abs((np.max(cons_vio_vec) - np.min(cons_vio_vec))/np.max(cons_vio_vec))
    cons_vio_diff = np.max(cons_vio_vec) - np.min(cons_vio_vec)

    if cons_vio_ratio <= 5 or cons_vio_diff <= 0.01:
        #..no significant gain in creating pure nodes... so pick the simple formula...
        id_ = np.argmin(obj_value_vec)
    else:
        #..there is some merit now in picking the one with better CV value...
        id_ = np.argmin(cons_vio_vec)

    if id_ == 0:
        return ind1
    else:
        return ind2


def select_best_of_two_cons_vio_ad(ind1, ind2):
    ind_list = [ind1, ind2]
    cons_vio_vec = np.array([ind1.cons_vio_ul, ind2.cons_vio_ul])
    obj_value_vec = np.array([ind1.fitness_ul, ind2.fitness_ul])

    if np.min(cons_vio_vec) <= 0:
        if np.max(cons_vio_vec) <= 0:#..both are feasible...
            return ind_list[np.argmin(obj_value_vec)]

    #...return the dominated solution...
    if (cons_vio_vec[0] <= cons_vio_vec[1]) and (obj_value_vec[0] <= obj_value_vec[1]):
        return ind1
    elif (cons_vio_vec[0] >= cons_vio_vec[1]) and (obj_value_vec[0] >= obj_value_vec[1]):
        return ind2

    #..Now both are non-dominated!!!.. check the trade-off...Eqn len should not be double...
    cons_vio_ratio = 100*np.abs((np.max(cons_vio_vec) - np.min(cons_vio_vec))/np.max(cons_vio_vec))
    cons_vio_diff = np.max(cons_vio_vec) - np.min(cons_vio_vec)

    advantage_in_cons = cons_vio_ratio/100
    cost_of_eqn_len = (np.max(obj_value_vec) - np.min(obj_value_vec))/np.max(obj_value_vec)

    if advantage_in_cons >= cost_of_eqn_len:
        #...return the one with least cons vio
        return ind_list[np.argmin(cons_vio_vec)]
    else:
        return ind_list[np.argmin(obj_value_vec)]


def select_best_of_two(ind1, ind2):
    """
    does binary tournament and returns the winner
    :param ind1:
    :param ind2:
    :return:
    """

    if ind1.cons_vio_ul == ind2.cons_vio_ul:
        if ind1.cons_vio_ul > 0:#...extreme rare event...
            if ind1.fitness_ul < ind2.fitness_ul:
                return ind1
            elif ind1.fitness_ul > ind2.fitness_ul:
                return ind2

            if ind1.abs_flag < ind2.abs_flag:
                return ind1
            elif ind1.abs_flag > ind2.abs_flag:
                return ind2
            else:
                r = np.random.rand()
                if r < 0.5:
                    return ind1
                else:
                    return ind2

    my_pop = [ind1, ind2]

    compute_net_fitness(my_pop)

    if my_pop[0].net_fitness < my_pop[1].net_fitness:
        winner = my_pop[0]
    elif my_pop[0].net_fitness > my_pop[1].net_fitness:
        winner = my_pop[1]
    else:#..both are having same fitness
        if my_pop[0].cons_vio_ul < my_pop[1].cons_vio_ul:
            winner = my_pop[0]
        elif my_pop[0].cons_vio_ul > my_pop[1].cons_vio_ul:
            winner = my_pop[1]
        else:
            if my_pop[0].abs_flag < my_pop[1].abs_flag:
                winner = my_pop[0]
            elif my_pop[0].abs_flag > my_pop[1].abs_flag:
                winner = my_pop[1]
            else:
                r = np.random.rand()
                if r < 0.5:
                    winner = my_pop[0]
                else:
                    winner = my_pop[1]

    return winner

