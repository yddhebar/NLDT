import numpy as np
import pandas as pd
import random

from iai_utils.factory_utils import fman
from iai_utils.factory_utils import fexp
from iai_utils.factory_utils import class_distribution
from iai_utils import decision_tree_funcs as dt_funcs
from iai_utils import iai_util_funcs as iai_funcs
from iai_utils import factory_utils
from main_codes.parameter_file import GlobalParameters

g_params = GlobalParameters()


class Rule:
    def __init__(self):
        self.abs_flag = []
        self.b_mat = []
        self.w = []
        self.bias = []


class Tree:

    def __init__(self,
                 max_depth=3,
                 min_size=10,
                 tau_impurity=0.01):
        self.max_depth = max_depth
        self.min_size = min_size
        self.tau_impurity = tau_impurity
        self.tree = {}

    def get_iai_split(self, data=[], tau_impurity=0.01, min_size=5, depth=None):
        """

        :param data: original data with class labels in the last row
        :param tau_impurity: Minimum Impurity of a node required to conduct a split
        :param min_size: Minimum number of datapoints in a node required to conduct a split
        :return: dictionary
        """

        gini_data = dt_funcs.compute_gini_score(data)
        class_labels = data[:, -1].astype(int)
        counts = np.bincount(class_labels)
        if (gini_data <= tau_impurity) or (data.shape[0] <= min_size) or depth >= self.max_depth:
            return {'node_id': None, 'node_impurity': gini_data, 'node_class': np.argmax(counts),
                    'node_type': 'leaf'}

        data_weights = None
        if g_params.impurity_measure == 'gini':
            features_array = data[:, :-1]
        elif g_params.impurity_measure == 'weighted_gini':
            features_array = data[:, :-2]
            data_weights = data[:, -2]

        rule = iai_funcs.determine_rule(features_array, class_labels, data_weights=data_weights)

        rule_satisfied_array = iai_funcs.get_rule_satisfaction_array(features_array, rule)
        left_data = data[rule_satisfied_array, :]
        right_data = data[~rule_satisfied_array, :]

        #... if the split is not doing enough split, we call this node as 'leaf'..
        if np.sum(rule_satisfied_array) < 5 or np.sum(~rule_satisfied_array) < 5:
            node_type = 'leaf'
        else:
            node_type = 'active'

        return {'node_id': None,
                'node_impurity': gini_data,
                'rule': rule,
                'node_class': np.argmax(counts),
                'node_type': node_type,# 'active',
                'split_impurity': dt_funcs.quality_of_split(left_data, right_data),
                'left_data': left_data,
                'right_data': right_data}

    def split(self, node, depth, max_depth=10, min_size=5, tau_impurity=0.01):
        """
        Conducts Recursive Splitting of the node of a decision-tree
        :param node:
        :param max_depth:
        :param min_size:
        :param depth:
        :return:
        """
        node.update({'depth': depth})
        if node['node_type'] == 'leaf':
            return

            # check for max depth
        if depth >= max_depth:
            # node['left'], node['right'] = to_terminal(left), to_terminal(right)
            node['node_type'] = 'leaf'
            return

        left, right = node['left_data'], node['right_data']
        del node['left_data']
        del node['right_data']
        del node['split_impurity']

        # process left child
        node['left_node'] = self.get_iai_split(left, tau_impurity=tau_impurity, min_size=min_size,
                                               depth=depth + 1)
        self.split(node['left_node'], depth + 1, max_depth, min_size, tau_impurity)

        node['right_node'] = self.get_iai_split(right, tau_impurity=tau_impurity, min_size=min_size,
                                                depth=depth + 1)
        self.split(node['right_node'], depth + 1, max_depth, min_size, tau_impurity)

    def train_tree(self, features, class_labels, max_depth=10, min_size=5,
                   tau_impurity=0.01,
                   class_dict=None,
                   training_data_share=None,
                   data_weights=None):

        self.tree = {}
        if class_dict is None:
            self.tree['class_id_dict'] = self.generate_class_id_dict(class_labels)
        else:
            self.tree['class_id_dict'] = class_dict

        self.tree['class_label_dict'] = self.generate_class_label_dict()
        self.max_depth = max_depth
        class_id_array = self.convert_class_labels_to_ids(class_labels)

        # ...Split the Data
        self.tree['data_ids'] = factory_utils.split_data(class_id_array, training_data_share)
        features = features[self.tree['data_ids']['train_ids'], :]
        class_id_array = class_id_array[self.tree['data_ids']['train_ids']]
        if g_params.impurity_measure == 'weighted_gini':
            data_weights = data_weights[self.tree['data_ids']['train_ids']]

        self.tree['n_features'] = features.shape[1]

        self.tree['X_min'] = np.min(features, axis=0)
        self.tree['X_max'] = np.max(features, axis=0)
        self.tree['mean'] = np.mean(features, axis = 0)
        self.tree['std'] = np.std(features, axis=0)
        n_features = features.shape[1]
        for i in range(n_features):
            if self.tree['X_min'][i] == self.tree['X_max'][i]:
                self.tree['X_min'][i] = 0
                self.tree['X_max'][i] = 1

        #features_nomralized = 1 + (features -
        #                           self.tree['X_min']) / (self.tree['X_max'] - self.tree['X_min'])
        features_nomralized = self.normalize_features(features)

        root_node = self.build_tree(features_nomralized, class_id_array,
                                    max_depth=max_depth,
                                    min_size=min_size,
                                    tau_impurity=tau_impurity,
                                    data_weights=data_weights)
        self.tree.update(root_node)
        self.data_distribution_in_tree_from_c_id(features_nomralized, class_id_array)
        self.write_text_to_tree()

    def build_tree(self, features, class_array, max_depth=10, min_size=5, tau_impurity=0.01,
                   data_weights=None):
        class_array = class_array.reshape([len(class_array), 1])
        if g_params.impurity_measure == 'gini':
            training_data = np.concatenate((features, class_array), axis=1)
        elif g_params.impurity_measure == 'weighted_gini':
            data_weights = data_weights.reshape([len(data_weights), 1])
            training_data = np.concatenate((features, data_weights, class_array), axis=1)

        root = self.get_iai_split(training_data, tau_impurity, min_size, depth=0)
        self.split(root, 0, max_depth, min_size, tau_impurity=tau_impurity)

        my_counter = 0
        total_number_of_nodes = dt_funcs.assign_address_to_nodes(root, my_counter)
        root.update({'total_nodes': total_number_of_nodes})

        dt_funcs.assign_class_labels_from_dict(root, self.tree['class_id_dict'])
        return root

    def data_distribution_in_tree_from_c_id(self, features, class_array):
        class_set = self.tree['class_id_dict']
        self.compute_data_distribution_in_tree(self.tree, features, class_array, class_set)

    def data_distribution_in_tree_from_c_label(self, features, class_array):
        class_array_c_id = self.convert_class_labels_to_ids(class_array)
        self.data_distribution_in_tree_from_c_id(features, class_array_c_id)

    def compute_data_distribution_in_tree(self, node, features, class_array, class_set):
        """
        assignes to each node how to data is splitted recursively to each node of the tree
        :param node:
        :param data:
        :return:
        """

        # ..Get Class Distribution...
        class_dist = class_distribution(class_array, class_set)
        node['class_dist'] = class_dist
        node['total_pts'] = features.shape[0]
        if node['node_type'] == 'leaf':
            return

        class_array = class_array.reshape((len(class_array), 1))
        node_data = np.concatenate((features, class_array), axis=1)

        node_split_bool_array = iai_funcs.get_rule_satisfaction_array(features, node['rule'])

        left_node_data = node_data[node_split_bool_array, :]
        right_node_data = node_data[~node_split_bool_array, :]

        node_left = node['left_node']
        node_right = node['right_node']

        self.compute_data_distribution_in_tree(node_left, features=left_node_data[:, :-1],
                                               class_array=left_node_data[:, -1], class_set=class_set)

        self.compute_data_distribution_in_tree(node_right, features=right_node_data[:, :-1],
                                               class_array=right_node_data[:, -1], class_set=class_set)

    def generate_class_id_dict(self, class_label_array):
        class_id_dict = {}
        c_id = 0
        unique_class_labels = pd.unique(class_label_array)
        for class_label in unique_class_labels:
            class_id_dict[c_id] = class_label
            c_id += 1

        return class_id_dict

    def generate_class_label_dict(self):
        class_id_dict = self.tree['class_id_dict']
        class_label_dict = {}
        for key, value in class_id_dict.items():
            class_label_dict[value] = key

        return class_label_dict

    def convert_class_labels_to_ids(self, class_label_series):
        """

        :param class_label_series: pandas series...
        :return: class_id_array: numpy array
        """
        class_id_array = -1 * np.ones(len(class_label_series))

        for key, value in self.tree['class_label_dict'].items():
            id_array = class_label_series == key
            class_id_array[id_array] = value

        return class_id_array

    def get_weights_from_tree(self):
        """
        recurcively obtain weights and biases to active nodes for the weight_array
        :return: weights_array: [w1,b1,w2,b2,..., wn, bn]
        """
        self.calculate_total_vars()
        node_type_array = dt_funcs.extract_node_type_array(self.tree)
        id_array = np.arange(self.tree['total_nodes'])
        active_node_id_array = id_array[node_type_array == 1]
        n_active_nodes = active_node_id_array.size
        n_weights_per_node = self.tree['rule'].w.size

        total_vars = self.tree['total_vars']
        weights_array = np.ones(total_vars) * np.nan
        active_node_id_array = self.tree['active_node_ids']

        location = 0
        for i in range(n_active_nodes):
            # location = i*(n_weights_per_node + n_bias_per_node)
            node = dt_funcs.extract_node_from_id(self.tree, active_node_id_array[i])
            n_bias = node['rule'].bias.size
            weights_array[location:location + n_weights_per_node] = node['rule'].w
            weights_array[location + n_weights_per_node:
                          location + n_weights_per_node + n_bias] = node['rule'].bias
            location += n_weights_per_node + 1 + node['rule'].abs_flag

        return weights_array

    def assign_weight_to_nodes_from_array(self, weights_array):
        """
        recurcively assings weights and biases to active nodes for the weight_array
        :param weights_array: [w1,b1,w2,b2,..., wn, bn]
        :param tree: decision tree
        :return:
        """

        node_type_array = dt_funcs.extract_node_type_array(self.tree)
        id_array = np.arange(self.tree['total_nodes'])
        active_node_id_array = id_array[node_type_array == 1]
        n_active_nodes = active_node_id_array.size

        n_weights_per_node = self.tree['rule'].w.size

        total_vars = self.tree['total_vars']
        active_node_id_array = self.tree['active_node_ids']

        if total_vars != len(weights_array):
            raise Exception('\ntotal_vars not equal to length of weight_array\n')

        location = 0
        for i in range(n_active_nodes):
            # location = i*(n_weights_per_node + n_bias_per_node)
            node = dt_funcs.extract_node_from_id(self.tree, active_node_id_array[i])
            n_bias = node['rule'].bias.size
            node['rule'].w = weights_array[location:location + n_weights_per_node]
            node['rule'].bias = weights_array[location + n_weights_per_node:
                                              location + n_weights_per_node + n_bias]
            location += n_weights_per_node + 1 + node['rule'].abs_flag

    def write_text_to_tree(self, detailed_text = True, colored_active_nodes = False):
        total_number_of_nodes = self.tree['total_nodes']

        for i in range(total_number_of_nodes):
            my_node = dt_funcs.extract_node_from_id(self.tree, i)
            self.write_node_text(my_node, detailed_text=detailed_text,
                                 colored_active_nodes = colored_active_nodes)

    def write_node_text(self, node, detailed_text = True, colored_active_nodes = False):
        temp_text = ''
        if node['node_type'] == 'active':
            if colored_active_nodes and node['node_id'] != 0:
                color_number = int(node['node_class']) % 9 + 1
                node['node_color'] = '/set39/' + str(color_number)  # str(int(node['node_class']) + 1)
            else:
                node['node_color'] = 'white'
            w = node['rule'].w
            bias = node['rule'].bias
            b_mat = node['rule'].b_mat
            n_prod_mat = b_mat.shape[0]
            n_vars = b_mat.shape[1]
            extra_bias = 0
            for i in range(n_prod_mat):
                prod_term_text = ''
                for j in range(n_vars):
                    e = int(b_mat[i, j])
                    if e != 0:
                        prod_term_text += '\\widehat{x_{' + str(j) + '}}'  # + '^{' + str(e) + '}'
                        if e != 1:
                            prod_term_text += '^{' + str(e) + '}'

                if prod_term_text != '':
                    # ..at least one exponent in power-law was non-zero..
                    w_m = fman(w[i])
                    w_e = fexp(w[i])
                    if abs(w_e) > 2:
                        w_text = str('\\left(%.1f \\times 10^{%d}\\right)' % (abs(w_m), w_e))
                    else:
                        w_text = str('%.2f ' % abs(w[i]))
                    if w[i] >= 0:
                        if len(temp_text) == 0:
                            temp_text += str(w_text)
                        else:
                            # temp_text += str(' + %.1e ' % w[i])
                            temp_text += str(' + ' + w_text)
                    else:
                        # temp_text += str(' - %.1e ' % abs(w[i]))
                        temp_text += str(' - ' + w_text)
                else:
                    extra_bias += w[i]

                temp_text += prod_term_text

            if node['rule'].abs_flag == 0:
                net_bias = bias + extra_bias
                bias_text = self.return_bias_text(net_bias)
            elif node['rule'].abs_flag == 1:
                net_bias = bias[0] + extra_bias
                bias_text = self.return_bias_text(net_bias)

            if net_bias >= 0:
                # temp_text += str(' + %.1e ' % net_bias)
                temp_text += str(' + ' + bias_text)
            else:
                # temp_text += str(' - %.1e ' % abs(net_bias))
                temp_text += str(' - ' + bias_text)

            if node['rule'].abs_flag == 1:
                net_bias = bias[1]
                bias_text = self.return_bias_text(net_bias)
                temp_text = '\\left|' + temp_text + '\\right| - ' + bias_text

            node['node_eqn_latex'] = temp_text

        else:

            color_number = int(node['node_class']) % 9 + 1

            node['node_color'] = '/set39/' + str(color_number)  # str(int(node['node_class']) + 1)
            '''
            if node['node_class'] == 1:
                node['node_color'] = 'lightblue'
            elif node['node_class'] == 2:
                node['node_color'] = 'yellow'
            '''

        if detailed_text:
            node_text = 'Node ' + str(node['node_id'])
            class_dist = node['class_dist']
            class_dist_text = '['
            for i in range(len(class_dist)):
                if i > 0:
                    class_dist_text += ', '
                class_dist_text += str(int(class_dist[i]))
            class_dist_text += ']'
            node_text += '\n' + class_dist_text
            node_text += '\n Class = ' + str(node['class_label'])
        else:
            node_text = str(node['node_id']) + '\n(' + str(node['class_label']) + ')'

        node.update({'node_text': node_text})

    def return_bias_text(self, net_bias):
        bias_exp = fexp(net_bias)
        bias_man = fman(net_bias)
        if abs(bias_exp) > 2:
            bias_text = str('\\left(%.1f \\times 10^{%d}\\right)' %
                            (abs(bias_man), bias_exp))
        else:
            bias_text = str('%.2f ' % abs(net_bias))
        return bias_text

    def calculate_total_vars(self):
        node_type_array = dt_funcs.extract_node_type_array(self.tree)
        id_array = np.arange(self.tree['total_nodes'])
        active_node_id_array = id_array[node_type_array == 1]
        self.tree['active_node_ids'] = active_node_id_array
        n_active_nodes = len(active_node_id_array)

        n_weights_per_node = len(self.tree['rule'].w)

        total_vars = n_weights_per_node * n_active_nodes

        # ..adding the bias term...
        for node_id in active_node_id_array:
            node = dt_funcs.extract_node_from_id(self.tree, node_id)
            total_vars += 1 + node['rule'].abs_flag

        # ..architecture of our search-array: [w1,b1,w2,b2,..,wn,bn]
        # total_vars = n_active_nodes * (n_weights_per_node + n_bias_per_node)
        self.tree['total_vars'] = total_vars
        # return total_vars

    def reassign_class_node(self, node):
        max_id = np.argmax(node['class_dist'])
        node['node_class'] = int(max_id)
        node['class_label'] = self.tree['class_label_dict'][max_id]
        if node['node_type'] == 'leaf':
            return
        else:
            self.reassign_class_node(node['left_node'])
            self.reassign_class_node(node['right_node'])

    def reassign_classes(self):
        self.reassign_class_node(self.tree)

    # ..Prediction....
    # ..Here we implement the vectorized version of doing prediction...
    # ...this is being done to speed up the prediction-process (it will be useful for global learning)...

    def assign_class_vectorised(self, node, data, id_array, class_array, b_space_flag=0):

        if node['node_type'] == 'leaf':
            class_array[id_array] = node['node_class']
            return class_array

        if b_space_flag == 0:
            rule_array = iai_funcs.get_rule_satisfaction_array(data[id_array, :], node['rule'])
        else:
            rule = node['rule']
            data_b_space = node['b_space_data']
            rule_array = iai_funcs.weighted_linear_split_rule(data_b_space[id_array, :],
                                                              rule.w, rule.bias, rule.abs_flag)

        id_array_left = id_array[rule_array]
        id_array_right = id_array[~rule_array]
        if len(id_array_left) > 0:
            class_array = self.assign_class_vectorised(node['left_node'], data, id_array_left, class_array,
                                                       b_space_flag)

        if len(id_array_right) > 0:
            class_array = self.assign_class_vectorised(node['right_node'], data, id_array_right, class_array,
                                                       b_space_flag)

        return class_array

    def predict_class_vectorized(self, testing_data, b_space_flag=0):
        """

        :param testing_data:
        :param b_space_flag: if this is 1, then we used the stored b-space-data of node,
        this is to boost up the prediction calculation...
        :return:
        """
        n_data_pts = testing_data.shape[0]
        id_array = np.arange(n_data_pts)
        class_array = np.ones(n_data_pts) * np.nan

        final_class_array = self.assign_class_vectorised(self.tree, testing_data,
                                                         id_array, class_array, b_space_flag)

        return final_class_array

    def store_b_space_data_in_node(self, node, data):
        #:param data: original X-space without class labels
        if node['node_type'] == 'leaf':
            return

        node['b_space_data'] = iai_funcs.transform_to_b_space(data, node['rule'].b_mat)
        self.store_b_space_data_in_node(node['left_node'], data)
        self.store_b_space_data_in_node(node['right_node'], data)

    def store_b_space_data_in_tree(self, data):
        """
        coputing b_space is a time consuming process as it involves exponents
        for repetitive tasks of obtainign b_space for fixed B-Mats in active nodes,
        we here store the b_mapped data once
        :param data: data in x-space...
        :return:
        """
        self.store_b_space_data_in_node(self.tree, data)

    def remove_b_space_data_in_node(self, node):
        if node['node_type'] == 'leaf':
            return

        del node['b_space_data']

    def remove_b_space_data_in_tree(self):
        """
        removes all b-space data from tree to save space..
        :return:
        """
        self.remove_b_space_data_in_node(self.tree)

    def choose_action(self, s, p_explore = -1):
        s_reshaped = s.reshape((1, len(s)))
        s_normalized = self.normalize_features(s_reshaped)
        action_id = int(self.predict_class_vectorized(s_normalized)[0])
        # action = int(self.tree['class_dict'][action_id])
        r = np.random.rand()
        if r >= p_explore:
            #....Greedy Action...
            action = int(self.tree['class_id_dict'][action_id])
        else:
            #...Exploratory Action...
            b_ = self.tree['class_id_dict'].copy()
            del b_[action_id]
            action = random.choice(list(b_.values()))

        return action

    def write_tree_table_latex(self, f_name):
        file = open(f_name, 'w')
        total_nodes = self.tree['total_nodes']
        x_min_text = '['
        x_max_text = '['
        n_features = self.tree['X_min'].size
        for i in range(n_features):
            x_min_text += str('%.2f, ' % self.tree['X_min'][i])
            x_max_text += str('%.2f, ' % self.tree['X_max'][i])

        x_min_text = x_min_text[:-2] + ']'
        x_max_text = x_max_text[:-2] + ']'

        file.write('\\begin{table}')
        file.write('\n\\setcellgapes{5pt}')
        file.write('\n\\makegapedcells')
        file.write('\n\\centering')
        file.write('\n\\caption{IAI Tree Equations. \n$x^{min}$ = ' + x_min_text +
                   ',\n $x^{max}$ = ' + x_max_text + '.}')
        file.write('\n\\begin{tabular}{|c|c|}\\hline')
        file.write('\n{\\bf Node} & {\\bf Rule}\\\\\\hline\n')

        for node_id in range(total_nodes):
            node = dt_funcs.extract_node_from_id(self.tree, node_id)
            if node['node_type'] == 'active':
                file.write(str(str(node_id) + ' & $' + node['node_eqn_latex']) + '$ \\\\\\hline \n')

        file.write('\\end{tabular}')
        file.write('\n\\label{tab:iai_tree}')
        file.write('\n\\end{table}')

        file.close()

    def normalize_features(self, X):
        #mean_ = np.tile(self.tree['mean'], (X.shape[0],1))
        #sigma_ = np.tile(self.tree['std'], (X.shape[0],1))
        #X_normalized = (X - mean_)/sigma_
        X_normalized = 1 + (X - self.tree['X_min']) / (self.tree['X_max'] - self.tree['X_min'])
        return X_normalized

    def get_tree_max_depth(self):
        total_nodes = self.tree['total_nodes']
        max_depth = 0
        for node_id in range(total_nodes):
            node = dt_funcs.extract_node_from_id(self.tree, node_id)
            if max_depth < node['depth']:
                max_depth = node['depth']

        self.tree['max_depth'] = max_depth

    def count_number_of_active_nodes(self):
        total_nodes = self.tree['total_nodes']
        n_active = 0
        for i in range(total_nodes):
            node = dt_funcs.extract_node_from_id(self.tree, i)
            if node['node_type'] == 'active':
                n_active += 1

        self.tree['n_active_nodes'] = n_active

    def compute_n_non_zero_eqn(self, b_mat, w):
        """
        :param b_mat:
        :param w:
        :return: number of active terms...
        """

        n_nonzero = 0
        n_prods = len(w)

        for i in range(n_prods):
            if w[i] != 0:
                n_nonzero += sum(b_mat[i, :] != 0)

        return n_nonzero

    def compute_eqn_stats(self):
        # ...computes equation stats....
        total_nodes = self.tree['total_nodes']
        self.count_number_of_active_nodes()
        n_eqn_len_array = np.ones(self.tree['n_active_nodes']) * np.nan

        active_counter = 0
        for i in range(total_nodes):
            node = dt_funcs.extract_node_from_id(self.tree, i)
            if node['node_type'] == 'active':
                n_eqn_len_array[active_counter] = self.compute_n_non_zero_eqn(node['rule'].b_mat,
                                                                              node['rule'].w)
                active_counter += 1

        self.tree['eqn_len_total'] = np.sum(n_eqn_len_array)
        self.tree['eqn_len_avg'] = np.mean(n_eqn_len_array)
