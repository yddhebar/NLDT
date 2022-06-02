import numpy as np
from decimal import Decimal

#...Extracting Exponent and mantissa...
def fexp(number):
    if not isinstance(number,float):
        number = number[0]
    (sign, digits, exponent) = Decimal(number).as_tuple()
    return len(digits) + exponent - 1

def fman(number):
    if not isinstance(number,float):
        number = number[0]
    return Decimal(number).scaleb(-fexp(number)).normalize()

def preprocess_class_labels(class_array_from_data):
    """
    Prepares Classification data for training and testing
    """
    # ...convert classes to integers...
    n_data_pts = len(class_array_from_data)
    class_set = set(class_array_from_data)
    class_dict = {}
    class_array = np.ones(n_data_pts) * (-1)
    c_id = 1
    for c in class_set:
        class_array[class_array_from_data == c] = c_id
        class_dict.update({c_id: c})
        c_id += 1

    class_array = class_array

    return {'preprocessed_class_labels':class_array,'class_dict':class_dict}


def class_distribution(class_array, class_set):
    """

    :param class_array: array of classes of each datapoint
    :param class_set: set of unique classes of the original dataset
    :return: class_dist: class distribution array
    """
    class_array = class_array.reshape(len(class_array))
    n_classes = len(class_set)

    class_dist = np.ones(n_classes)*-1
    #... c_ids start from 1...
    for i in range(n_classes):
        #class_i_data = class_array[class_array.astype(int) == (i+1)]
        class_i_data = class_array[class_array.astype(int) == i]
        class_dist[i] = len(class_i_data)

    return class_dist


def compute_accuracy(my_tree, X, Y):
    res = preprocess_class_labels(Y)
    training_data_Y = res['preprocessed_class_labels']

    prediction_array = my_tree.predict_class_id(X)

    prediction_array = prediction_array.astype(int)
    actual_labels = Y.astype(int)
    diff_array = prediction_array - actual_labels
    accuracy = 100 * np.sum(diff_array == 0) / len(actual_labels)

    print('prediction accuracy = ', accuracy)


def split_data(class_array = None, training_data_share = None):
    '''

    :param class_ids: numpy array of class ids (0 or 1)
    :param training_data_share: percentage of training data
    :return: ids of training_datapts and testing_datapts
    '''

    # ...Classes should be 0 and 1....
    n_data_pts = class_array.size
    data_ids = np.arange(n_data_pts)
    unique_classes = np.unique(class_array)
    training_data_id_list = []
    testing_data_id_list = []
    #return {'train_ids': data_ids,
    #        'test_ids': data_ids}

    for c_ in unique_classes:
        class_c_ids = data_ids[class_array == c_]
        n_class_c = class_c_ids.size
        shuffled_c_ids = np.random.permutation(np.arange(n_class_c))
        n_training_c = int(np.round(n_class_c * training_data_share))
        training_c_ids = class_c_ids[shuffled_c_ids[:n_training_c]]
        testing_c_ids = class_c_ids[shuffled_c_ids[n_training_c:]]
        training_data_id_list.append(training_c_ids)
        testing_data_id_list.append(testing_c_ids)


    training_data_ids = np.concatenate(training_data_id_list)
    testing_data_ids = np.concatenate(testing_data_id_list)

    return {'train_ids':training_data_ids,
            'test_ids': testing_data_ids}


def split_data_binary_class(class_array = None, training_data_share = None):
    '''
    Application exclusively for only binary classification problems...
    :param class_ids: numpy array of class ids (0 or 1)
    :param training_data_share: percentage of training data
    :return: ids of training_datapts and testing_datapts
    '''

    # ...Classes should be 0 and 1....
    n_data_pts = class_array.size
    data_ids = np.arange(n_data_pts)

    class_a_ids = data_ids[class_array == 0]
    class_b_ids = data_ids[class_array == 1]

    # ...Class a data prep....
    n_class_a = class_a_ids.size
    shuffled_a_ids = np.random.permutation(np.arange(n_class_a))
    n_training_a = int(np.round(n_class_a * training_data_share))
    training_a_ids = class_a_ids[shuffled_a_ids[:n_training_a]]
    testing_a_ids = class_a_ids[shuffled_a_ids[n_training_a:]]

    # ..Class b data prep....
    n_class_b = class_b_ids.size
    shuffled_b_ids = np.random.permutation(np.arange(n_class_b))
    n_training_b = int(np.round(n_class_b * training_data_share))
    training_b_ids = class_b_ids[shuffled_b_ids[:n_training_b]]
    testing_b_ids = class_b_ids[shuffled_b_ids[n_training_b:]]

    training_data_ids = np.concatenate([training_a_ids, training_b_ids])
    testing_data_ids = np.concatenate([testing_a_ids, testing_b_ids])

    return {'train_ids':training_data_ids,
            'test_ids': testing_data_ids}