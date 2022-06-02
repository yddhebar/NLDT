import numpy as np


def Compute_PID(x_array):
    #..input: states: numpy array of x in chronological order..
    #...returns P, I and D values...
    n = x_array.__len__()
    if n < 2:
        print('Value Error: n cannot be less than 2!')
        raise ValueError

    x_array_t = x_array[0:n-1]
    x_array_t1 = x_array[1:n]
    diff_array = x_array[-1] - x_array[-2]
    mean_sum_array = (x_array_t + x_array_t1)/2

    #...computing P, I and D..
    P = x_array[-1]
    I = mean_sum_array.sum()*(n-1)#..trapezoidal rule for numerical integration...
    D = diff_array.sum()

    return P,I,D


def Choose_Action_simple(w,b,state):
    x, v, theta, omega = state

    f_value = w[0]*theta + w[1]*omega + b

    if f_value <= 0:
        action = 1
    else:
        action = 0

    return action


def Choose_Action(w_array,b_array,states):
    #..states: numpy matrix of states... each row is one state...
    #..returns action, whether to go left or right...

    n_state_vars = states.shape[1]
    pid_controller_array = np.zeros(n_state_vars)

    for i in range(n_state_vars):
        s_array = states[:,i]
        y, integral_y, dy_dt = Compute_PID(s_array)
        w = w_array[:, i]  # ..first column of w_array....
        b = b_array[i]
        pid_controller_array[i] = w[0] * y + w[1] * integral_y + w[2] * dy_dt + b

    f_value = pid_controller_array.sum()

    if f_value <= 0:
        action = 0
    else:
        action = 1
    return action
