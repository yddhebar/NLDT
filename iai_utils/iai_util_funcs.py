import numpy as np
import math
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.factory import get_termination
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.core.sampling import Sampling

from iai_utils import decision_tree_funcs as dt_funcs
from iai_ga.iai_ga_classes import Indiv
from iai_ga.iai_ga import bilevel_ga
from main_codes.parameter_file import GlobalParameters

g_params = GlobalParameters()

from scipy.optimize import minimize as minimize_scipy


class SplitQualityProblem(Problem):

    def __init__(self, n_var=2, data=[] , abs_flag = 0):
        super().__init__(n_var=n_var,
                         n_obj=1,
                         n_constr=0,
                         xl=np.array(n_var*[-1]),
                         xu=np.array(n_var*[1]))
        self.data = data
        self.abs_flag = abs_flag

    def _evaluate(self, x, out, *args, **kwargs):
        pop_size = x.shape[0]
        f_values = np.zeros(pop_size)
        if self.abs_flag == 1:
            w = x[:,:-2]
            b = x[:,-2:]
        else:
            w = x[:,:-1]
            b = x[:,-1]
        for i in range(pop_size):
            w_temp = w[i,:]
            if self.abs_flag == 1:
                b_temp = b[i,:]
            else:
                b_temp = b[i]

            f_values[i] = compute_split_score(data=self.data,
                                              w=w_temp,
                                              bias=b_temp,
                                              abs_flag=self.abs_flag)
            #f_values[i] = -Evaluate_Performance(w_array,b[i,:],max_iter)

        out["F"] = f_values


class MySampling(Sampling):
    """
    Randomly sample points in the real space by considering the lower and upper bounds of the problem.
    """

    def __init__(self, var_type=float) -> None:
        super().__init__()
        self.var_type = var_type

    def _do(self, problem, n_samples, **kwargs):
        #val = np.random.random((n_samples, problem.n_var))
        n_vars = problem.n_var
        #..load problem data...
        data = problem.data
        class_labels = data[:,-1].astype(int)

        if g_params.impurity_measure == 'gini':
            features_array = data[:,:-1]
        elif g_params.impurity_measure == 'weighted_gini':
            features_array = data[:,:-2]

        n_data_pts = len(class_labels)
        unique_classes = set(class_labels)
        n_classes = len(unique_classes)
        r1 = np.random.randint(n_classes)
        r2 = np.random.randint(n_classes)
        if r1 == r2:
            r2 = r1-1

        id_array = np.arange(n_data_pts)

        init_pop = np.zeros((n_samples,problem.n_var))

        for i in range(n_samples):
            w_bias_norm = get_w_bias_using_dipole(features_array, class_labels, id_array)
            w = w_bias_norm[:-1]
            bias = w_bias_norm[-1]

            if problem.abs_flag == 0:
                init_pop[i,:-1] = w
                init_pop[i,-1] = -bias
            elif problem.abs_flag == 1:
                init_pop[i, :-2] = w
                init_pop[i, -2] = -bias
                init_pop[i,-1] = np.random.rand()

        return init_pop


def get_w_bias_using_dipole(features_array, Y_values,id_array):

    #...check if it is even possible to get dipoles...
    min_array = np.min(features_array, axis = 0)
    max_array = np.max(features_array, axis = 0)
    if np.all(min_array == max_array):
        #...not possible to get the dipole!!!....
        n_w = features_array.shape[1]
        w_bias_norm = 1 - 2*np.random.random(n_w + 1)
        return w_bias_norm


    while True:
        a_id, b_id = np.random.choice(id_array, 2)
        x_a = features_array[a_id, :]
        x_b = features_array[b_id, :]
        w = x_a - x_b
        delta = 0.5#np.random.rand()
        bias = delta * (np.dot(w, x_a)) + (1 - delta) * (np.dot(w, x_b))

        dipole_check = (np.dot(w,x_a) - bias)*(np.dot(w,x_b) - bias)
        if dipole_check > 0:
            raise Exception('Invalid Dipole = %f !!!' % dipole_check)

        if dipole_check < 0:
            break

    # ...Normalize w and bias...
    w_bias = np.concatenate((w, np.array([bias])))
    w_bias_norm = w_bias / max(abs(w_bias))
    return w_bias_norm


def get_w_bias_using_dipole_old_faulty(features_array,class_labels,id_array):
    unique_classes = set(class_labels)
    n_classes = len(unique_classes)

    while True:
        a_id, b_id = pick_xa_xb_ids(n_classes, class_labels, id_array)
        x_a = features_array[a_id, :]
        x_b = features_array[b_id, :]
        w = x_a - x_b
        delta = 0.5#np.random.rand()
        bias = delta * (np.dot(w, x_a)) + (1 - delta) * (np.dot(w, x_b))

        dipole_check = (np.dot(w,x_a) - bias)*(np.dot(w,x_b) - bias)
        if dipole_check > 0:
            raise Exception('Invalid Dipole = %f !!!' % dipole_check)

        if dipole_check < 0:
            break

    # ...Normalize w and bias...
    w_bias = np.concatenate((w, np.array([bias])))
    w_bias_norm = w_bias / max(abs(w_bias))
    return w_bias_norm


def pick_xa_xb_ids(n_classes, class_labels, id_array):
    """
    pics two ids of datapoints belonging to different class
    :param n_classes: number of unique classes in the given dataset
    :param class_labels: class labels of datapoints in the dataset
    :param id_array: id array (representing indices) of datapoints in a dataset
    :return:
    """
    r1 = np.random.randint(n_classes)
    r2 = np.random.randint(n_classes)
    if r1 == r2:
        r2 = r1 - 1
    unique_class_labels = list(set(class_labels))
    c1_id_array = id_array[class_labels == unique_class_labels[r1]]
    c2_id_array = id_array[class_labels == unique_class_labels[r2]]

    n_c1 = len(c1_id_array)
    n_c2 = len(c2_id_array)

    a_id = c1_id_array[np.random.randint(n_c1)]
    b_id = c2_id_array[np.random.randint(n_c2)]

    return a_id, b_id


def cross_entropy_loss(data, w, bias, abs_flag = 0):
    #...Computes the Cross-Entropy Loss...
    """
    :param data: data with class id in the last column...
    :param w_vec: w and bias
    :param abs_flag:
    :return: split quality (delta impurity)
    """
    features = data[:, :-1]
    #....The First Weight is for division....
    w_true = w[1:]
    rule_values = get_weighted_sum_values(features, w_true, bias,
                                          abs_flag=abs_flag)
    eps = 1e-15
    if w[0] < eps:
        w[0] = eps
    rule_values = rule_values/w[0]

    #....compute Sigmoid....
    y_hat = sigmoid(rule_values)
    y = data[:,-1]#... has only 0 and 1....
    n_classes = np.unique(y).size

    #..cross entropy loss..
    y_hat[y_hat < eps] = eps#.... to avoid log(0)...
    y_hat[y_hat > 1 - eps] = 1 - eps#..to avoid log(0)...
    if n_classes != 2:
        #raise Exception('Cross Entropy loss is only allowed for N-classes = 2')
        c_loss = cross_entropy_multi_class(y,y_hat)
    else:
        #....Reconfigure y-values so that y is either 0 OR 1...
        y_unique = np.unique(y)
        y_ = 1*y#...create a copy...
        y[y_ == y_unique[0]] = 0
        y[y_ == y_unique[1]] = 1
        c_loss_vec = y*np.log(y_hat) + (1 - y)*np.log(1 - y_hat)
        c_loss = -np.sum(c_loss_vec)

    return c_loss/y.size


def cross_entropy_multi_class(y, y_hat):
    #...Computes cross-entropy loss for multi-class...
    y_unique = np.unique(y)#...Unique classes...
    n_classes = y_unique.size

    p_sum_array = np.zeros([n_classes, 2])
    '''
        this is for the binary split, 
        each row will indicate if the datapoint is 
        near y_hat = 1 or y_hat = 0....
    '''

    for i in range(n_classes):
        c_value = y_unique[i]
        p_sum_array[i,0] = np.sum(np.log(y_hat[y == c_value]))
        p_sum_array[i,1] = np.sum(np.log(1 - y_hat[y == c_value]))

    #p_max_array = np.max(p_sum_array, axis = 1)
    #...identify the best class-id corresponding to y_hat = 1 and y_hat = 0...
    c_id_yhat_0 = np.argmax(p_sum_array[:,0])
    c_id_yhat_1 = np.argmax(p_sum_array[:,1])

    s = p_sum_array[c_id_yhat_0, 0] + p_sum_array[c_id_yhat_1,1]

    #...pruning out rows corresponding to above class ids...
    p_sum_filtered = np.delete(p_sum_array, (c_id_yhat_0, c_id_yhat_1), axis=0)
    #s += np.sum(np.max(p_sum_filtered, axis=1))
    c_loss = -s

    return c_loss


def sigmoid(x):
    tau = 100
    x[x <= -tau] = -tau #... to avoid exponents of large numbers...
    return 1/(1 + np.exp(-x))


def compute_split_score(data, w, bias, abs_flag = 0):
    """
    :param data: data with class id in the last column...
    :param w_vec: w and bias
    :param abs_flag:
    :return: split quality (delta impurity)
    """

    if g_params.impurity_measure == 'gini':
        features = data[:, :-1]
    elif g_params.impurity_measure == 'weighted_gini':
        features = data[:, :-2]

    rule_satisfied_array = weighted_linear_split_rule(features,
                                                      w, bias,
                                                      abs_flag=abs_flag)
    left_data = data[rule_satisfied_array, :]
    right_data = data[~rule_satisfied_array, :]
    split_score = dt_funcs.quality_of_split(left_data, right_data)

    return split_score


def transform_to_b_space(data, B_mat):
    """
    Transforms data from X-space to B-space based on the entries in B_mat matrix..
    :param data: only features, no classes
    :param B_mat: B Matrix of exponents
    :return: transformed coordinates in B-space...
    """
    n_power_laws = B_mat.shape[0]
    n_data_pts = data.shape[0]
    x_transform = np.ones((n_data_pts, n_power_laws))

    for j in range(n_power_laws):
        power_value_array = data**B_mat[j,:]
        x_transform[:,j] = power_value_array.prod(axis = 1)

    return x_transform


def get_rule_value(features, rule):
    """
    :param data: only features....(without class labels)
    :param B_mat: B Matrix of exponents
    :param w: weights of power law
    :param b: bias (or biases)
    :param abs_flag: absolute flag..
    :return: split-rule value given by w*x + b
    """
    b_space = transform_to_b_space(features, rule.b_mat)
    w = rule.w
    b = rule.bias
    abs_flag = rule.abs_flag
    w_reshaped = w.reshape((len(w),1))
    if abs_flag == 0:
        rule_value_array = np.matmul(b_space,w_reshaped) + b
    else:
        rule_value_array = abs(np.matmul(b_space, w_reshaped) + b[0]) - abs(b[1])

    return rule_value_array


def get_weighted_sum_values(data, w, b, abs_flag = 0):
    """
    :param data: only features....(without class labels)
    :param B_mat: B Matrix of exponents
    :param w: weights of power law
    :param b: bias (or biases)
    :param abs_flag: absolute flag..
    :return: weighted sum values of the rule given by w*x + b <= 0
    """

    w_reshaped = w.reshape((len(w), 1))
    if abs_flag == 0:
        rule_value_array = np.matmul(data, w_reshaped) + b
    else:
        rule_value_array = abs(np.matmul(data, w_reshaped) + b[0]) - abs(b[1])

    return rule_value_array.flatten()


def weighted_linear_split_rule(data, w, b, abs_flag = 0):
    """
    :param data: only features....(without class labels)
    :param B_mat: B Matrix of exponents
    :param w: weights of power law
    :param b: bias (or biases)
    :param abs_flag: absolute flag..
    :return: boolean array of which data-point satisfies the split-rule given by w*x + b <= 0
    """

    w_reshaped = w.reshape((len(w),1))
    if abs_flag == 0:
        rule_value_array = np.matmul(data,w_reshaped) + b
    else:
        rule_value_array = abs(np.matmul(data, w_reshaped) + b[0]) - abs(b[1])

    rule_satisfied_array = rule_value_array[:,0] <= 0

    return rule_satisfied_array


def determine_weights_and_biases_rga(X, Y, abs_flag=0, data_weights = None):
    """
    uses RGA to determine those weights for which best split is obtained...
    :param data: with class labels in the last column...
    :param abs_flag:
    :return:
    """
    if abs_flag == 0:
        n_vars = X.shape[1] + 1
    elif abs_flag == 1:
        n_vars = X.shape[1] + 2

    if g_params.impurity_measure == 'gini':
        data = np.concatenate((X, Y.reshape(len(Y),1)), axis=1)
    elif g_params.impurity_measure == 'weighted_gini':
        data = np.concatenate((X, data_weights.reshape(len(Y),1), Y.reshape(len(Y),1)), axis = 1)
    problem = SplitQualityProblem(n_var=n_vars, data=data, abs_flag = abs_flag)
    #....running GA...
    termination = get_termination("f_tol", tol=0.01, n_last=5, n_max_gen=50, nth_gen=5)

    algorithm = GA(
        pop_size=40,
        eliminate_duplicates=True,
        sampling=MySampling())

    res = minimize(problem,
                   algorithm,
                   termination=termination,#('n_gen', 100),
                   verbose=False)
    #print('Done with determining weights and biases:')
    #print('W = ',res.X[:-1], ' bias = ',res.X[-1])
    return res.X


def determine_weights_and_biases_nelder_pymoo(X, Y, abs_flag=0):
    """
    uses RGA to determine those weights for which best split is obtained...
    :param data: with class labels in the last column...
    :param abs_flag:
    :return:
    """
    n_features = X.shape[1]
    if abs_flag == 1:
        n_vars = n_features + 2
    else:
        n_vars = n_features + 1

    data = np.concatenate((X, Y.reshape(len(Y),1)), axis=1)
    problem = SplitQualityProblem(n_var=n_vars, data=data, abs_flag = abs_flag)
    #....running GA...
    termination = get_termination("f_tol", tol=0.01, n_last=5, n_max_gen=50, nth_gen=5)

    # ..initial pt..
    id_array = np.arange(len(Y))
    init_pts = []
    for _ in range(10):
        w_0 = get_w_bias_using_dipole(X, Y, id_array)
        init_pts.append(w_0)

    algorithm = NelderMead(X=np.row_stack(init_pts),
                           n_max_local_restarts=10)

    res = minimize(problem,
                   algorithm,
                   termination=termination,#('n_gen', 100),
                   verbose=False)
    #print('Done with determining weights and biases:')
    #print('W = ',res.X[:-1], ' bias = ',res.X[-1])
    return res.X


def determine_weights_and_biases_classical(X, Y, abs_flag=0):
    """
    uses classical optimization algorithm to determine weights and biases
    :param X:
    :param Y:
    :param abs_flag:
    :return:
    """
    #..initial pt..

    id_array = np.arange(len(Y))
    w_bias_norm = get_w_bias_using_dipole(X,Y, id_array)

    w = w_bias_norm[:-1]
    bias = w_bias_norm[-1]

    if abs_flag == 0:
        n_vars = X.shape[1] + 1
        w_0 = np.ones(n_vars)*np.nan
        w_0[:-1] = w
        w_0[-1] = -bias
    elif abs_flag == 1:
        n_vars = X.shape[1] + 2
        w_0 = np.ones(n_vars) * np.nan
        w_0[:-2] = w
        w_0[-2] = -bias
        w_0[-1] = np.random.rand()

    lower_bounds = np.ones([len(w_0),1])*-1
    lower_bounds[-1] = 0
    if g_params.loss_function_ll == 'cross_entropy':
        lower_bounds = np.concatenate([[[0]], lower_bounds])
        w_0 = np.concatenate([[np.random.rand()], w_0])
    upper_bounds = np.ones([len(w_0),1])
    bounds = np.concatenate([lower_bounds, upper_bounds], axis = 1)

    res = minimize_scipy(fun = classical_obj_func,
                         x0 = w_0,
                         args = (X,Y, abs_flag),
                         method = 'SLSQP',
                         bounds=bounds,)

    if g_params.loss_function_ll == 'cross_entropy':
        w_final = res.x[1:]
    else:
        w_final = res.x

    #print('C_loss = %.2e' % res.fun)
    return w_final


def classical_obj_func(w_all, X, Y, abs_flag):
    data = np.concatenate([X, Y.reshape([len(Y),1])], axis = 1)

    if abs_flag == 0:
        w = w_all[:-1]
        bias = w_all[-1]
    elif abs_flag == 1:
        w = w_all[:-2]
        bias = w_all[-2:]
    if g_params.loss_function_ll == 'cross_entropy':
        obj_func_1 = 0

        obj_func_1 = cross_entropy_loss(data=data,
                                       w=w,
                                       bias=bias,
                                       abs_flag=abs_flag)


        obj_func_2 = compute_split_score(data=data,
                                       w=w[1:],
                                       bias=bias,
                                       abs_flag=abs_flag)
        #obj_func = (1*obj_func_1 + 9*obj_func_2)/10
        obj_func = obj_func_1
        #obj_func = obj_func_2
    else:
        obj_func = compute_split_score(data = data,
                                       w=w,
                                       bias=bias,
                                       abs_flag=abs_flag)


    return obj_func


def determine_weights_and_biases(X, Y, abs_flag=0, data_weights = None):

    if g_params.loss_function_ll == 'cross_entropy':
        w = determine_weights_and_biases_classical(X, Y, abs_flag)
    else:
        if g_params.ll_algorithm == 'classical':
            w = determine_weights_and_biases_classical(X, Y, abs_flag)
        elif g_params.ll_algorithm == 'rga':
            w = determine_weights_and_biases_rga(X,Y,abs_flag, data_weights)

    #w = determine_weights_and_biases_nelder_pymoo(X, Y, abs_flag)

    return w


def determine_rule(X,Y, data_weights = None):

    rule = bilevel_ga(X,Y, data_weights=data_weights)

    return rule


def get_rule_satisfaction_array(data, rule):
    """

    :param data: original X-space without class labels
    :param rule: rule object
    :return: rule satisfaction array 1 means rule is satisfied, 0 means rule is not satisfied
    """
    b_space = transform_to_b_space(data, rule.b_mat)

    rule_satisfaction_array = weighted_linear_split_rule(b_space, rule.w, rule.bias, rule.abs_flag)

    return rule_satisfaction_array
