"""
frechet_mean of positive matrices w.r.t Fisher-Rao metric.
"""

import numpy as np
import scipy.linalg
import logging

logging.basicConfig(level=logging.INFO)


def fisher_distance_s(p1,p2):
    '''
    The Fisher distance between two positive definite matrices : ||log(p1^-1p2)||_F
    ||_F is the Frobnius  norm
    :param p1: Posdef matrix
    :param p2: Posdef matrix
    :return: distance between p1 and p2
    '''
    sqr_p1 = scipy.linalg.sqrtm(p1)
    in_sqr_p1 = np.linalg.inv(sqr_p1)
    log_mat = scipy.linalg.logm(in_sqr_p1 @ p2 @ in_sqr_p1)
    return np.linalg.norm(log_mat)**2

def cost_fisher(positive_matrices, x):
    '''
    The cost function for GD algorithm
    :param positive_matrices:
    :param x: current mean
    :return: Sum(|x-p_i|^2)
    '''
    cost = 0
    for p in positive_matrices:
        cost+=fisher_distance_s(p,x)
    return cost


def is_pos_def(x):
    return np.linalg.norm(x-x.T) < 1e-8 and np.all(np.linalg.eigvals(x) > 0)

def frechet_mean(positive_matrices):
    """
    Calculate the Frechet Mean according with Graditent Descent Algorithm.
    Implementation can be found in
    "Computing the Karcher mean of symmetric positive definite matrices. D.A. Bini 2013"
    :param positive_matrices: positive definite nxn matrices
    :return: the Freche Mean of the matrices
    """

    for idx, p in enumerate(positive_matrices):
        if p.dtype!=np.dtype('float64') and p.dtype!=np.dtype('float32'):
            raise Exception("only support float matrices")
        if not is_pos_def(p):
            raise Exception(f"matrix {idx} is not positive definite")

    max_learning_rate = 0.1
    learning_rate = max_learning_rate
    num_iters = 100
    stop_cond_diff = 10**-10
    mean = positive_matrices[1]
    inv_matrices = [np.linalg.inv(p) for p in positive_matrices]

    cost = cost_fisher(positive_matrices, mean)
    logging.info(f"fisher cost before {cost}")

    for i in range(num_iters):
        cost = cost_fisher(positive_matrices, mean)
        logging.debug(f"fisher cost is {cost} lr {learning_rate}")

        new_mean = np.copy(mean)
        sqrt_m = scipy.linalg.sqrtm(mean)

        for inv_p,p in zip(inv_matrices, positive_matrices):
            log_term = scipy.linalg.logm(sqrt_m@inv_p @sqrt_m )
            new_mean -= learning_rate*sqrt_m @log_term@sqrt_m

        new_cost = cost_fisher(positive_matrices, new_mean)

        if not is_pos_def(new_mean) or new_cost > cost:
            learning_rate = learning_rate/2.0
        else:
            diff = np.linalg.norm(new_mean-mean)
            learning_rate = min(max_learning_rate, learning_rate*1.5)
            mean = np.copy(new_mean)

    cost = cost_fisher(positive_matrices, mean)
    logging.info(f"fisher cost after {cost}")

    return mean
