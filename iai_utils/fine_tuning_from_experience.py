#..Finetuning the parametes of decision tree by interacting with the environment...
import numpy as np

from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_termination
from pymoo.model.problem import Problem
from pymoo.optimize import minimize
from iai_utils.decision_tree_funcs import compute_total_reward
import multiprocessing as mp
from functools import partial
from pymoo.model.sampling import Sampling


class TotalRewardProblem(Problem):

    def __init__(self,n_var=2, my_tree = {}, env = [], parallel_flag = 1, p_explore = -1):
        super().__init__(n_var=n_var,
                         n_obj=1,
                         n_constr=0,
                         xl=np.array(n_var*[-1]),
                         xu=np.array(n_var*[1]))
        #....parallel_flag can be eigher 0 or 1 to indicate if parallization is required...
        self.my_tree = my_tree
        self.env = env
        self.parallel_flag = parallel_flag
        self.p_explore = p_explore

    def _evaluate(self, x, out, *args, **kwargs):
        pop_size = x.shape[0]
        f_values = np.zeros(pop_size)

        if self.parallel_flag == 0:
            for i in range(pop_size):
                w_array = x[i,:]
                f_values[i] = evaluate_obj(w_array, self.my_tree, self.env, p_explore = self.p_explore)
        elif self.parallel_flag == 1:
            #print('Parallel Evaluation...')
            n_cpus = np.min([50, mp.cpu_count()])
            if n_cpus <= 0:
                Exception('n_cpus (%d) <= 0 !!' % n_cpus)
            #print('N processors: ', n_cpus)
            f_partial = partial(evaluate_obj, my_tree = self.my_tree,
                                env = self.env, p_explore = self.p_explore)
            #..convert to list
            w_list = [x[i,:] for i in range(pop_size)]
            with mp.Pool(processes=n_cpus) as pool:
                results = pool.map(f_partial, w_list)

            f_values = np.array(results)

        out["F"] = f_values


class MySampling(Sampling):
    """
    Randomly sample points in the real space by considering the lower and upper bounds of the problem.
    """

    def __init__(self, var_type=np.float) -> None:
        super().__init__()
        self.var_type = var_type

    def _do(self, problem, n_samples, **kwargs):
        #val = np.random.random((n_samples, problem.n_var))
        n_vars = problem.n_var
        x_L = problem.xl
        x_U = problem.xu

        w_tree = problem.my_tree.get_weights_from_tree()

        init_pop = np.zeros((n_samples,problem.n_var))

        for i in range(n_samples):
            init_pop[i,:] = x_L + np.random.random(n_vars)*(x_U - x_L)

        #...seed the best set of weights in the pool...
        init_pop[0,:] = w_tree

        return init_pop


def evaluate_obj(weight_array, my_tree, env, p_explore=-1):
    my_tree.assign_weight_to_nodes_from_array(weight_array)

    obj_value = -compute_total_reward(my_tree, env, n_episodes= 20, max_iters= 500, p_explore=p_explore)

    return obj_value


def fine_tune_dt(my_tree = {}, env = [], p_explore = -1):

    my_tree.calculate_total_vars()
    n_vars = my_tree.tree['total_vars']
    problem = TotalRewardProblem(n_var=n_vars, my_tree = my_tree, env= env, p_explore = p_explore)
    # ....running GA...
    termination = get_termination("f_tol", tol=0.00001, n_last=10, n_max_gen=30, nth_gen=1)

    algorithm = GA(
        pop_size=50,#5*n_vars,
        eliminate_duplicates=True,
        save_history=True,
    sampling=MySampling())

    res = minimize(problem,
                   algorithm,
                   termination=termination,  # ('n_gen', 100),
                   verbose=True)

    print('Done with fine tuning')

    weights_array = res.X

    my_tree.assign_weight_to_nodes_from_array(weights_array)

    return my_tree


