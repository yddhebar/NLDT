import numpy as np
from iai_utils import iai_util_funcs as iai_funcs


def fine_tune_best_indiv(params, best_ind, X_train, Y_train, data_weights = None):
    x_transformed = iai_funcs.transform_to_b_space(X_train, best_ind.b_mat)
    w_all = iai_funcs.determine_weights_and_biases_rga(x_transformed, Y_train, best_ind.abs_flag, data_weights)
    # w_all = np.random.random(params.n_prod_terms + 1)
    # w_all = res[0]
    # f_ll_best = res[1]

    if best_ind.abs_flag == 0:
        w = w_all[:-1]
        bias = w_all[-1]
    elif best_ind.abs_flag == 1:
        w = w_all[:-2]
        bias = w_all[-2:]

    best_ind.w = w
    best_ind.bias = bias

    if data_weights is None:
        combined_array = np.concatenate((x_transformed,
                                         Y_train.reshape((len(Y_train), 1))),
                                        axis=1)
    else:
        combined_array = np.concatenate((x_transformed,
                                         data_weights.reshape((len(Y_train), 1)),
                                         Y_train.reshape((len(Y_train), 1))),
                                        axis=1)
    best_ind.fitness_ll = iai_funcs.compute_split_score(data=combined_array,
                                                      w=w,
                                                      bias=bias,
                                                      abs_flag=best_ind.abs_flag)

    best_ind.cons_vio_ul = best_ind.fitness_ll - params.tau_impurity

    return best_ind