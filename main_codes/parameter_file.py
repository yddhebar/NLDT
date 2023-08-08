class GlobalParameters():
    def __init__(self):
        self.device = 'cpu'
        self.n_prod_terms = 3
        self.a_max = None#...only one term can appear...
        self.tau_impurity = 0.05
        self.selection_method = 'normal'#'cons_ratio_ad'#'cons_ratio'#
        self.impurity_measure = 'gini'#'weighted_gini'#
        self.loss_function_ll = 'gini'#'cross_entropy'#
        self.ll_algorithm = 'classical'#'rga'#
