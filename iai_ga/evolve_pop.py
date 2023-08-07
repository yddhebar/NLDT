import numpy as np
import copy
from iai_ga.evaluate_pop import compute_net_fitness
from iai_ga.selection import selection
from iai_ga.cross_over import cross_over_pop
from iai_ga.mutation import mutate_pop
from iai_ga.repair_pop import repair_pop


def evolve_pop(pop, params):

    parent_pop = copy.deepcopy(pop)
    compute_net_fitness(parent_pop)
    parent_pop = selection(parent_pop, params)
    child_pop = cross_over_pop(pop=parent_pop, params=params)
    child_pop = mutate_pop(child_pop, params)
    child_pop = repair_pop(child_pop, params)

    return child_pop

