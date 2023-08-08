import numpy as np

def extract_vector(pop, attr_name):
    """
    Extract vector of a given "attr_name" for entire population...
    :param pop:
    :param attr_name:
    :return:
    """

    pop_size = len(pop)
    e_vec = np.ones(pop_size)*-1

    for i in range(pop_size):
        e_vec[i] = getattr(pop[i], attr_name)

    return e_vec
