"""
%...repair prod_mat....
%..if all entries in prodmat are 0, then randomly make one entry non-zero(using the list of allowable powers)....
%...removes redundancy in the prod_mat....
"""

import numpy as np
import random

def repair_pop(pop, params):
    #check for copy and deepcopy stuff ... this is python.....
    random.seed(params.seed)
    np.random.seed(params.seed)
    pop_size = len(pop)
    for i in range(pop_size):
        prod_mat_ind = pop[i].b_mat
        prod_mat_ind = repair_ind(prod_mat_ind, params)
        prod_mat_ind = change_redundant_rows_to_zero(prod_mat_ind)
        prod_mat_ind = set_to_max_active_terms(prod_mat_ind, params.a_max)
        pop[i].b_mat = prod_mat_ind

    pop = fix_duplicates_local(pop, params)

    return pop


def set_to_max_active_terms(prod_mat, max_active_terms):
    prod_mat_repaired = prod_mat
    n_rows = prod_mat.shape[0]
    n_cols = prod_mat.shape[1]
    col_ids = np.array(list(range(n_cols)))

    for i in range(n_rows):
        prod_mat_row = prod_mat[i,:]
        active_ids = col_ids[prod_mat_row != 0]
        n_active = len(active_ids)
        if n_active <= max_active_terms:
            continue
        else:
            n_diff = n_active - max_active_terms
            random.shuffle(active_ids)#...shuffle active ids...
            for j in range(n_diff):
                prod_mat_repaired[i, active_ids[j]] = 0

    return prod_mat_repaired


def repair_ind(prod_mat, params):
    """
    repairs prod_mat by making one of its element as non-zero...
    :param prod_mat:
    :param params:
    :return:
    """
    zero_mat = np.zeros(prod_mat.shape)
    n_rows = zero_mat.shape[0]
    n_cols = zero_mat.shape[1]
    allowable_non_zero_powers = params.allowable_powers[params.allowable_powers != 0]
    if np.array_equal(prod_mat, zero_mat):
        i = np.random.randint(n_rows)
        j = np.random.randint(n_cols)

        k = np.random.randint(len(allowable_non_zero_powers))
        prod_mat[i,j] = allowable_non_zero_powers[k]

    return prod_mat


def change_redundant_rows_to_zero(prod_mat):
    n_rows = prod_mat.shape[0]
    n_cols = prod_mat.shape[1]
    zero_row = np.zeros(n_cols)

    for i in range(n_rows):
        ref_row = prod_mat[i,:]
        for j in range(i+1,n_rows):
            comp_row = prod_mat[j,:]
            if np.array_equal(comp_row, zero_row):
                continue

            if np.array_equal(comp_row, ref_row):
                prod_mat[j,:] = zero_row

    return prod_mat


def fix_duplicates_local(pop, params):
    """
    checks for duplicates in the current pop. if duplicates exist, try to
    re-mutate that individual, until its unique. Also ensure that all the
    population members are unique.
    :param pop:
    :param params:
    :return:
    """
    pop_size = len(pop)
    n_rows = pop[0].b_mat.shape[0]
    n_cols = pop[0].b_mat.shape[1]
    n_allowable_powers = len(params.allowable_powers)

    for i in range(1, pop_size):
        p = pop[i]
        prod_mat_ind = p.b_mat
        local_archive = pop[0:i-1]

        while(1):
            if p not in local_archive:
                #..duplicate of p is not in local_archive...
                break

            rand_row_id = np.random.randint(n_rows)
            rand_col_id = np.random.randint(n_cols)
            rand_pow_id = np.random.randint(n_allowable_powers)
            prod_mat_ind[rand_row_id, rand_col_id] = params.allowable_powers[rand_pow_id]
            prod_mat_ind = repair_ind(prod_mat_ind, params)
            prod_mat_ind = change_redundant_rows_to_zero(prod_mat_ind)
            prod_mat_ind = set_to_max_active_terms(prod_mat_ind, params.a_max)

        pop[i].b_mat = prod_mat_ind

    return pop
