# ..Finetuning the parametes of decision tree by interacting with the environment...
import numpy as np

from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_termination
from pymoo.model.problem import Problem
from pymoo.optimize import minimize
from iai_utils.decision_tree_funcs import compute_accuracy_from_c_id
from pymoo.model.sampling import Sampling


class TotalAccuracyProblem(Problem):

    def __init__(self, n_var=2, my_tree={}, X_data=None, Y_data=None, b_space_flag=None):
        super().__init__(n_var=n_var,
                         n_obj=1,
                         n_constr=0,
                         xl=np.array(n_var * [-1]),
                         xu=np.array(n_var * [1]))
        self.my_tree = my_tree
        self.X_data = X_data
        self.Y_data = Y_data
        self.b_space_flag = b_space_flag

    def _evaluate(self, x, out, *args, **kwargs):
        pop_size = x.shape[0]
        f_values = np.zeros(pop_size)
        for i in range(pop_size):
            w_array = x[i, :]

            f_values[i] = evaluate_obj(w_array, self.my_tree, self.X_data, self.Y_data, self.b_space_flag)

        out["F"] = f_values


# ...seed the best set of weights in the pool...
class MySampling(Sampling):
    """
    Randomly sample points in the real space by considering the lower and upper bounds of the problem.
    """

    def __init__(self, var_type=np.float) -> None:
        super().__init__()
        self.var_type = var_type

    def _do(self, problem, n_samples, **kwargs):
        # val = np.random.random((n_samples, problem.n_var))
        n_vars = problem.n_var
        x_L = problem.xl
        x_U = problem.xu

        w_tree = problem.my_tree.get_weights_from_tree()

        init_pop = np.zeros((n_samples, problem.n_var))

        for i in range(n_samples):
            init_pop[i, :] = x_L + np.random.random(n_vars) * (x_U - x_L)

        # ...seed the best set of weights in the pool...
        init_pop[0, :] = w_tree

        return init_pop


def evaluate_obj(weight_array, my_tree, X_data, Y_data, b_space_flag=0):
    my_tree.assign_weight_to_nodes_from_array(weight_array)

    obj_value = -compute_accuracy_from_c_id(my_tree, X_data, Y_data, b_space_flag=b_space_flag)

    return obj_value


def fine_tune_dt(my_tree=None, X_data=None, Y_data=None, b_space_flag=None):
    my_tree.calculate_total_vars()
    total_vars = my_tree.tree['total_vars']

    n_vars = total_vars

    # ..Normalize Features...
    X_data = my_tree.normalize_features(X_data)

    # ..Store b-space features in each node for fast execution...
    print('Store B-space data in each node...')
    my_tree.store_b_space_data_in_node(my_tree.tree, X_data)
    b_space_flag = 1
    print('Done storing B-space data in each node...')

    problem = TotalAccuracyProblem(n_var=n_vars, my_tree=my_tree,
                                   X_data=X_data, Y_data=Y_data, b_space_flag=b_space_flag)
    # ....running GA...
    termination = get_termination("f_tol", tol=0.01, n_last=10, n_max_gen=100, nth_gen=1)

    algorithm = GA(
        pop_size=10 * n_vars,
        eliminate_duplicates=True,
        save_history=True,
        sampling=MySampling())

    print('Starting Fine Tuning...')
    res = minimize(problem,
                   algorithm,
                   termination=termination,  # ('n_gen', 100),
                   verbose=True)

    print('Done with fine tuning')

    weights_array = res.X

    my_tree.assign_weight_to_nodes_from_array(weights_array)

    return my_tree
