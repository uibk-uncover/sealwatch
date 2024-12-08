"""Implementation of sample-pair analysis (SPA).

From Dumitrescu, Wu, Memon:
On Steganalysis of Random LSB Embedding in Continuous-tone Images.
https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=540b2431fa24e8602dddbc3f48eda262950c466e

Inspired by implementation of Remi Cogranne.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np


def attack(
    x0: np.ndarray,
) -> float:
    """Run sample-pair analysis.

    :param cover_spatial:
    :type cover_spatial: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: embedding rate estimate
    :rtype: float

	:Example:

    >>> spatial = np.array(Image.open('suspicious.png'))
    >>> alpha_hat = sw.spa.attack(spatial)
    >>> assert alpha_hat == 0
    """
    # get sample pairs
    u = x0[:, :-1].flatten().astype('int')
    v = x0[:, 1:].flatten().astype('int')

    # construct primary sets
    P = np.ones(u.size)
    X = (v % 2) & (u > v) | ~(v % 2) & (u < v)
    Y = (v % 2) & (u < v) | ~(v % 2) & (u > v)
    Z = (u == v)
    W = ((u//2) == (v//2)) & ~Z
    V = Y & ~W

    # solve quadratic equation for q
    gamma = W.sum() + Z.sum()
    coef = [
        gamma/2,
        2*X.sum() - P.sum(),
        Y.sum() - X.sum(),
    ]
    q = np.roots(coef)

    # select and modify the root
    q = np.real(q)  # real part (if complex)
    q = min(q)  # the smaller root
    if q < .01:  # negatives / very low estimate
        q = 0.
    return q
