#..definations of functions related to decision tree...
import numpy as np
from scipy.optimize import minimize_scalar
import copy
from iai_multiclass_classification.main_codes.parameter_file import GlobalParameters

g_params = GlobalParameters()


def compute_gini_score_regular(data):
    # ...computes gini score....
    # ..data is a numpy array, with each row indicating one datapoint
    # ...the last row of 'data' is the class index...
    class_values = data[:, -1]
    n_data = len(class_values)
    unique_classes = np.unique(class_values)

    gini_value = 1

    for c in unique_classes:
        n_c = len(class_values[class_values == c])
        gini_value -= (n_c / n_data) ** 2

    return gini_value


def compute_gini_score_weighted(data):
    # ...computes weighted gini score....
    # ..data is a numpy array, with each row indicating one datapoint
    # ...the last column of 'data' is the class index...
    #...the second last column are "weight_values"...
    class_values = data[:, -1]
    data_weights = data[:,-2]
    n_data = np.sum(data_weights)
    unique_classes = np.unique(class_values)

    gini_value = 1

    for c in unique_classes:
        n_c = np.sum(np.array(class_values == c)*data_weights)
        gini_value -= (n_c / n_data) ** 2

    return gini_value


def compute_gini_score(data):
    # ...computes gini score....
    # ..data is a numpy array, with each row indicating one datapoint
    # ...the last row of 'data' is the class index...

    if g_params.impurity_measure == 'gini':
        gini_value = compute_gini_score_regular(data)
    elif g_params.impurity_measure == 'weighted_gini':
        gini_value = compute_gini_score_weighted(data)

    return gini_value


def quality_of_split(left_data, right_data):
    """
    :param left_data: left node data (numpy array)
    :param right_data: right node data (numpy arrray)
    :return: weighted gini-score of left and right node
    lower the value of split_quality, better the split...
    """

    if g_params.impurity_measure == 'gini':
        n_left = left_data.shape[0]
        n_right = right_data.shape[0]
    elif g_params.impurity_measure == 'weighted_gini':
        n_left = np.sum(left_data[:,-2])
        n_right = np.sum(left_data[:,-2])

    left_gini = compute_gini_score(left_data)
    right_gini = compute_gini_score(right_data)

    split_quality = (n_left * left_gini + n_right * right_gini) / (n_left + n_right)

    return split_quality


def assign_address_to_nodes(node, counter):
    """
    This function assignes address nodes of a decision tree
    :param node:
    :param counter:
    :return:
    """
    if node['node_id'] is not None: #!= 0:
        return counter
    elif node['node_type'] == 'leaf':
        node['node_id'] = counter
        counter += 1
        return counter
    else:
        node['node_id'] = counter
        counter += 1
        counter = assign_address_to_nodes(node['left_node'], counter)
        counter = assign_address_to_nodes(node['right_node'], counter)
        return counter


def extract_node_from_id(node, node_id):
    """
    This will be genrally used for post-processing
    :param node: Root node of the decision tree
    :param node_id: address of the node
    :return: node whose address is "node_id"
    """
    if node['node_id'] == node_id:
        return node
    elif node['node_type'] == 'leaf':
        return None
    else:
        r = extract_node_from_id(node['left_node'], node_id)
        if r == None:
            r = extract_node_from_id(node['right_node'], node_id)
        return r


def assign_class_labels_from_dict(node, class_dict):
    #...assigs class label to a node from its corresponding class_id (given by node['node_class']...
    node['class_label'] = class_dict[node['node_class']]

    if node['node_type'] == 'leaf':
        return

    left_node = node['left_node']
    right_node = node['right_node']

    assign_class_labels_from_dict(left_node, class_dict)
    assign_class_labels_from_dict(right_node, class_dict)


def extract_node_type_array(tree):
    """
    boolean array of which node is active and leaf
    1 == Active Node, 0 == Leaf Node
    :param tree: decision tree
    :return:
    """
    n_nodes = tree['total_nodes']
    node_type_array = np.ones(n_nodes)*(-1)

    for i in range(n_nodes):
        node = extract_node_from_id(tree, i)

        if node['node_type'] == 'active':
            node_type_array[i] = 1
        else:
            node_type_array[i] = 0

    return node_type_array


def convert_class_label_to_class_id(tree, Y_labels):
    class_dict = tree['class_id_dict']
    Y_id = []

    for c_label in Y_labels:
        for c in class_dict:
            if class_dict[c] == c_label:
                Y_id.append(c)
                break

    return np.array(Y_id)


def compute_accuracy_from_c_id(my_tree, X, Y_id, b_space_flag = 0):
    """
    computes accuracy of predicting X features
    :param my_tree: Tree class
    :param X:
    :param Y_id: class id (not class label)
    :return:
    """
    if b_space_flag == 0:
        prediction_array = my_tree.predict_class_vectorized(X)
    else:
        prediction_array = my_tree.predict_class_vectorized(X,b_space_flag)

    prediction_array = prediction_array.astype(int)
    diff_array = prediction_array - Y_id
    accuracy = 100 * np.sum(diff_array == 0) / len(Y_id)

    return accuracy


def compute_accuracy_from_c_label(my_tree, X, Y, b_space_flag = 0):
    """
    computes accuracy of predicting X features
    :param my_tree: Tree class
    :param X:
    :param Y: class label (not class_id)
    :return:
    """

    #Y_id = convert_class_label_to_class_id(my_tree.tree, Y)
    Y_id = my_tree.convert_class_labels_to_ids(Y)

    accuracy = compute_accuracy_from_c_id(my_tree, X, Y_id, b_space_flag)

    return accuracy


def compute_total_reward(my_tree, env, n_episodes = 1, max_iters = 1000, p_explore = -1):
    """

    :param tree: Trained RL IAI Tree...
    :param max_iter: maximum number of iterations before episode terminates
    :return: total reward collected
    """
    rewards_array = np.zeros(n_episodes)
    for e in range(n_episodes):
        #print("Episode %d" % (e + 1))
        s = env.reset()
        ep_r = 0
        for i in range(max_iters):
            # print(f'iteration = {i}')
            action = int(my_tree.choose_action(s, p_explore=p_explore))
            s_, r_, done, info = env.step(action)
            ep_r += r_
            if done:
                break
            s = s_

        rewards_array[e] = ep_r

    #env.close()

    return np.mean(rewards_array)


def construct_pruned_tree(node, tau_split_data = 5):

    node['node_id'] = ''
    if node['node_type'] == 'leaf':
        return

    data_dist_left = node['left_node']['class_dist']
    data_dist_right = node['right_node']['class_dist']

    net_data_split = min(sum(data_dist_left), sum(data_dist_right))

    if net_data_split <= tau_split_data:
        del node['left_node']
        del node['right_node']
        node['node_type'] = 'leaf'
    else:
        construct_pruned_tree(node['left_node'], tau_split_data)
        construct_pruned_tree(node['right_node'], tau_split_data)


def pruned_tree_for_depth(my_tree, max_depth = None):
    my_pruned_tree = copy.deepcopy(my_tree)
    total_nodes = my_tree.tree['total_nodes']

    my_pruned_tree.tree['node_id'] = None#..setting node id of the "root_node" to None for recounting..

    #..reset Node ids..
    for node_id in range(1,total_nodes):
        node = extract_node_from_id(my_pruned_tree.tree, node_id)
        if node is None:
            continue
        node['node_id'] = None

    my_pruned_tree.tree = prune_node_to_depth(my_pruned_tree.tree, max_depth)

    my_pruned_tree.tree['total_nodes'] = assign_address_to_nodes(my_pruned_tree.tree, 0)

    return my_pruned_tree


def prune_node_to_depth(node, max_depth):

    if node['depth'] >= max_depth:
        if node['node_type'] == 'leaf':
            return node
        #..else the node is an active node...
        #...in this situation make it a leaf node..
        node['node_type'] = 'leaf'
        #..delete all the child nodes..
        del node['left_node']
        del node['right_node']

        return node
    else:
        if node['node_type'] == 'leaf':
            return node
        else:
            node['left_node'] = prune_node_to_depth(node['left_node'], max_depth)
            node['right_node'] = prune_node_to_depth(node['right_node'], max_depth)

            return node


def prune_tree_for_accuracy(my_tree, features, class_labels, tau_accuracy = 0.5, min_data_pts = None):
    Y_id = my_tree.convert_class_labels_to_ids(class_labels)
    X = np.array(features)
    total_accuracy = compute_accuracy_from_c_id(my_tree, X, Y_id)
    my_pruned_tree = copy.deepcopy(my_tree)

    total_nodes = my_tree.tree['total_nodes']

    my_pruned_tree.tree['node_id'] = None#..setting node id of the "root_node" to None for recounting..

    #...Prune only when tau_accuracy > 0....
    if tau_accuracy > 0:
        for node_id in range(1,total_nodes):
            temp_tree = copy.deepcopy(my_pruned_tree.tree)
            node = extract_node_from_id(my_pruned_tree.tree, node_id)

            if node is None:
                continue

            if node['node_type'] == 'active':
                # ...make this node as leaf and trim everything else that follows..
                node['node_type'] = 'leaf'
                del node['left_node']
                del node['right_node']
                pruned_accuracy = compute_accuracy_from_c_id(my_pruned_tree, X, Y_id)

                #..if the pruned accuracy falls way too much, then don't prune it..
                if total_accuracy - pruned_accuracy > tau_accuracy:
                    my_pruned_tree.tree = temp_tree

    #..reset Node ids..
    for node_id in range(1,total_nodes):
        node = extract_node_from_id(my_pruned_tree.tree, node_id)
        if node is None:
            continue
        node['node_id'] = None

    my_pruned_tree.tree = remove_redundant_nodes_with_less_data(my_pruned_tree.tree, min_data_pts)

    my_pruned_tree.tree = remove_redundant_split_to_leaf_nodes(my_pruned_tree.tree)
    #..do this second time..(actually we have to do it until all redundant splits_to_leaf are removed...
    my_pruned_tree.tree = remove_redundant_split_to_leaf_nodes(my_pruned_tree.tree)

    my_pruned_tree.tree['total_nodes'] = assign_address_to_nodes(my_pruned_tree.tree, 0)

    return my_pruned_tree


def prune_tree(my_tree, tau_data_split = 5):
    #..prune tree....
    my_pruned_tree = copy.deepcopy(my_tree)
    tree = my_pruned_tree.tree
    construct_pruned_tree(tree, tau_data_split)

    my_counter = 0
    total_number_of_nodes = assign_address_to_nodes(tree, my_counter)
    tree['total_nodes'] = total_number_of_nodes

    my_pruned_tree.tree = tree

    return my_pruned_tree

def remove_redundant_nodes_with_less_data(node, min_data_pts = None):
    #...also removes if a node has less than "min_data_pts" number of points...
    node['node_id'] = None
    if node['node_type'] == 'leaf':
        return node

    n_data_left_node = np.sum(node['left_node']['class_dist'])
    n_data_right_node = np.sum(node['right_node']['class_dist'])

    if n_data_left_node < min_data_pts and n_data_right_node < min_data_pts:
        node['node_type'] = 'leaf'
        del node['left_node']
        del node['right_node']
    elif n_data_left_node < min_data_pts:
        if node['right_node']['node_type'] == 'active':
            node['right_node'] = remove_redundant_nodes_with_less_data(node['right_node'], min_data_pts)
        temp_node = copy.deepcopy(node['right_node'])
        del node['left_node']
        del node['right_node']
        node.update(temp_node)
    elif n_data_right_node < min_data_pts:
        if node['left_node']['node_type'] == 'active':
            node['left_node'] = remove_redundant_nodes_with_less_data(node['left_node'], min_data_pts)
        temp_node = copy.deepcopy(node['left_node'])
        del node['left_node']
        del node['right_node']
        node.update(temp_node)
    else:
        #...both child nodes have > min_data_pts...
        node['left_node'] = remove_redundant_nodes_with_less_data(node['left_node'], min_data_pts)
        node['right_node'] = remove_redundant_nodes_with_less_data(node['right_node'], min_data_pts)

    return node


def remove_redundant_split_to_leaf_nodes(node):
    #...remove the split if it results to leaf nodes of same class...
    if node['node_type'] == 'leaf':
        return node

    if node['left_node']['node_type'] == 'leaf' and node['right_node']['node_type'] == 'leaf':
        if node['left_node']['node_class'] == node['right_node']['node_class']:
            node['node_type'] = 'leaf'
            del node['left_node']
            del node['right_node']
            return node

    node['left_node'] = remove_redundant_split_to_leaf_nodes(node['left_node'])
    node['right_node'] = remove_redundant_split_to_leaf_nodes(node['right_node'])

    return node


def convert_decision_tree_to_list(my_tree):
    total_nodes = my_tree['total_nodes']

    tree_list = total_nodes*[None]

    for i in range(total_nodes):
        node = extract_node_from_id(my_tree, i)
        node_in_list = copy.deepcopy(node)
        if node['node_type'] == 'active':
            node_in_list['left_node_id'] = node_in_list['left_node']['node_id']
            node_in_list['right_node_id'] = node_in_list['right_node']['node_id']
            del node_in_list['left_node']
            del node_in_list['right_node']

        tree_list[i] = node_in_list

    return tree_list





