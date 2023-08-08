import numpy as np
import copy
from iai_ga.extract_vector import extract_vector


def merge_and_reduce_pop(parent_pop, child_pop):
    """
    filters out top pop_size population members (does mu + lambda survival selection)
    :param parent_pop:
    :param child_pop:
    :return: filtered pop
    """
    pop_size = len(parent_pop)

    parent_pop_unique = extract_unique_indivs(parent_pop)

    merged_pop = parent_pop_unique + child_pop

    net_fitness_vec = extract_vector(merged_pop, 'net_fitness')

    sorted_ids = np.argsort(net_fitness_vec)

    filtered_pop = [merged_pop[x] for x in sorted_ids[0:pop_size]]

    return copy.deepcopy(filtered_pop)


def extract_unique_indivs(pop):
    """

    :param pop:
    :param params:
    :return: unique_pop
    """
    pop_size = len(pop)
    unique_pop = []

    for i in range(pop_size):
        p = pop[i]

        if p not in unique_pop:
            #..duplicate of p is not in local_archive...
            unique_pop.append(p)

    return unique_pop