"""
List of objective functions
"""
import numpy as np


def fancy_hinge(y_true, y_pred, slope=0, flip=False, transition=0):
    """
    This is a regression cost function that penalises overestimation much more harshly than underestimation
    (and visa versa)

    :param y_true:
    :param y_pred:
    :param slope: slope of the less harsh penalty
    :param flip: if True then the underestimation is heavily penalised
    :param transition: where to switch from the linear to the polynomial cost
    :return:
    """
    delta = y_true - y_pred

    above, below = delta > transition, delta <= transition
    if flip:
        below, above = above, below

    delta[above] = delta[above] ** 2
    delta[below] = np.abs(delta[below] * slope)

    return delta.sum(axis=0)  # Sum columns
