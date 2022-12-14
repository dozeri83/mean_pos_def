"""
Test Frechet mean against analytic geometric mean
"""
from frechet_mean_pos_def import *

import numpy as np
import scipy.linalg
import logging


def generate_random_positive_matrix(dim):
    l = np.random.random(size=(dim, dim))
    return l@l.T


def generate_random_positive_diagonal_matrix(dim):
    p = np.zeros(shape=( dim, dim))
    for i in range(dim):
        p[i][i]= np.random.uniform(0,100)

    return p


def geometric_mean_analytic(a1,a2):
    """
    See "A Review of Geometric Mean of Positive Definite Matrices", Wen-Haw Chen
    :param a1: positive definite matrix of dimention n
    :param a2:positive definite matrix of dimention n
    :return: gedometric average of a1 and a2
    """
    sqrt_a1 = scipy.linalg.sqrtm(a1)
    inv_sqrt_a1  = np.linalg.inv(sqrt_a1)
    m = inv_sqrt_a1@a2@inv_sqrt_a1
    sqrt_m = scipy.linalg.sqrtm(m)

    return sqrt_a1@sqrt_m@sqrt_a1


def geometric_mean_diagonals_analytic(diagonal_matrices):

    num_matrices = len(diagonal_matrices)
    dim = diagonal_matrices[0].shape[0]
    multiplication = np.eye(dim)
    geometric_mean = np.eye(dim)

    for  p in diagonal_matrices:
        multiplication = multiplication@ p
    for i in range(dim):
        geometric_mean[i][i] = np.power(multiplication[i][i], 1.0/num_matrices)

    return geometric_mean



def test_frechet_mean_geometric_average_two_matrices():
    np.random.seed(8128)
    num_tests=10
    for i in range(num_tests):
        dim = np.random.randint(2,10)
        a1 = generate_random_positive_matrix(dim)
        a2 = generate_random_positive_matrix(dim)

        gm_analytic = geometric_mean_analytic(a1, a2)
        gm_numerical = frechet_mean([a1,a2])


        assert np.linalg.norm(gm_analytic - gm_numerical) < 1e-5

def test_frechet_mean_geometric_average_diagonals():
    np.random.seed(28)
    num_tests=10
    for i in range(num_tests):
        dim = np.random.randint(2,10)
        num_matrices = np.random.randint(2,30)
        positive_matrices = []
        for i in range(num_matrices):
            positive_matrices.append(generate_random_positive_diagonal_matrix(dim))
        gm_numeric = frechet_mean(positive_matrices)
        gm_analytic = geometric_mean_diagonals_analytic(positive_matrices)

        assert np.linalg.norm(gm_numeric - gm_analytic) < 1e-5
