"""Deflection coefficient of the likelihood ratio test (LRT)
under the multivariate Gaussian (MVG) cover model.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np
from typing import Tuple, Union

from ._variance import estimate_variance


def assumed_rates(
    ps: Tuple[np.ndarray],
    ps_hat: Union[Tuple[np.ndarray], str, float] = None,
) -> np.ndarray:
    """Returns the change rates assumed by the attacker.

    :param ps: actual change rates (p_p1, p_m1)
    :type ps: tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param ps_hat: assumed change rates,
        None for omniscient attacker (ps),
        'indifferent' for a constant rate (average of ps),
        a scalar for a given constant rate, or
        a tuple (p_p1, p_m1) for mismatched rates
    :type ps_hat: None, str, float or tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: assumed change rates of the same shape as ps
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> ps_hat = sw.mvg.assumed_rates(ps, 'indifferent')
    """
    ps = np.asarray(ps, dtype='float64')
    if ps_hat is None:
        return ps
    if isinstance(ps_hat, str):
        if ps_hat != 'indifferent':
            raise NotImplementedError(f'unknown ps_hat {ps_hat}')
        ps_hat = np.mean(ps)  # per direction
    return np.broadcast_to(np.asarray(ps_hat, dtype='float64'), ps.shape)


def deflection(
    x0: np.ndarray,
    ps: Tuple[np.ndarray],
    *,
    ps_hat: Union[Tuple[np.ndarray], str, float] = None,
    method: str = 'taylor',
    **kw,
) -> float:
    """Deflection coefficient of the likelihood ratio test (LRT)
    with cover x0 and change rates ps.

    The method assumes a Gaussian cover model with known variances.
    In practice, it is used to quickly benchmark steganographic methods,
    without need to train an actual classifier.

    It was introduced in
    Sedighi, Cogranne, Fridrich.
    Content-Adaptive Steganography by Minimizing Statistical Detectability.
    IEEE TIFS, 2016.

    :param x0: uncompressed (pixel) cover image
        of shape [height, width]
    :type x0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param ps: actual change rates (p_p1, p_m1)
    :type ps: tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param ps_hat: change rates assumed by the attacker,
        omniscient (ps) by default, see :func:`assumed_rates`
    :type ps_hat: None, str, float or tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param method: 'taylor' for the small-rate approximation (MiPOD),
        'exact' for the exact moments
    :type method: str
    :return: value of the deflection coefficient
    :rtype: float

    :Example:

    >>> rhos = cl.hill.compute_cost_adjusted(x0)
    >>> ps, _ = cl.simulate.probability(rhos, alpha=.4)
    >>> sw.mvg.deflection(x0, ps)  # omniscient
    >>> sw.mvg.deflection(x0, ps, ps_hat='indifferent')
    >>> sw.mvg.deflection(x0, ps, method='exact')
    """
    if method == 'taylor':
        return deflection_taylor(x0, ps, ps_hat=ps_hat, **kw)
    elif method == 'exact':
        return deflection_exact(x0, ps, ps_hat=ps_hat, **kw)
    raise NotImplementedError(f'unknown method {method}')


def deflection_taylor(
    x0: np.ndarray,
    ps: Tuple[np.ndarray],
    *,
    ps_hat: Union[Tuple[np.ndarray], str, float] = None,
    **kw,
) -> float:
    """Deflection coefficient of the LRT, Taylor approximation for small rates.

    The rates split into a symmetric part beta = (p_p1 + p_m1) / 2
    with Fisher information I_beta = 2 / sigma^4,
    and an asymmetric part gamma = (p_p1 - p_m1) / 2 (mean shift)
    with Fisher information I_gamma = 4 / sigma^2, both orthogonal.
    Then

    delta = sum(I_beta beta_hat beta + I_gamma gamma_hat gamma)
        / sqrt(sum(I_beta beta_hat^2 + I_gamma gamma_hat^2)),

    i.e., Eq. 11 for the omniscient and Eq. 12 for the indifferent attacker
    for symmetric rates.

    :param x0: uncompressed (pixel) cover image
        of shape [height, width]
    :type x0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param ps: actual change rates (p_p1, p_m1)
    :type ps: tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param ps_hat: assumed change rates, see :func:`assumed_rates`
    :type ps_hat: None, str, float or tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: value of the deflection coefficient
    :rtype: float
    """
    p_p1, p_m1 = np.asarray(ps, dtype='float64')
    ph_p1, ph_m1 = assumed_rates(ps, ps_hat)
    beta, gamma = (p_p1 + p_m1) / 2, (p_p1 - p_m1) / 2
    beta_hat, gamma_hat = (ph_p1 + ph_m1) / 2, (ph_p1 - ph_m1) / 2
    s2 = estimate_variance(x0, **kw)
    fi_beta, fi_gamma = 2. / s2**2, 4. / s2  # Fisher information
    num = np.sum(fi_beta * beta_hat * beta + fi_gamma * gamma_hat * gamma)
    den = np.sqrt(np.sum(fi_beta * beta_hat**2 + fi_gamma * gamma_hat**2))
    return float(num / den)


def deflection_exact(
    x0: np.ndarray,
    ps: Tuple[np.ndarray],
    *,
    ps_hat: Union[Tuple[np.ndarray], str, float] = None,
    nodes: int = 40,
    **kw,
) -> float:
    """Deflection coefficient of the LRT, without approximation.

    The attacker uses the log-likelihood ratio log q_hat / f,
    q_hat = (1 - p_p1 - p_m1) f(y) + p_p1 f(y-1) + p_m1 f(y+1) with the assumed rates.
    Its moments under cover and stego are computed by Gauss-Hermite quadrature.
    Matches :func:`deflection_taylor` for small rates.

    :param x0: uncompressed (pixel) cover image
        of shape [height, width]
    :type x0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param ps: actual change rates (p_p1, p_m1)
    :type ps: tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param ps_hat: assumed change rates, see :func:`assumed_rates`
    :type ps_hat: None, str, float or tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param nodes: number of quadrature nodes
    :type nodes: int
    :return: value of the deflection coefficient
    :rtype: float
    """
    p_p1, p_m1 = np.asarray(ps, dtype='float64')
    ph_p1, ph_m1 = assumed_rates(ps, ps_hat)
    log_ph0 = np.log1p(-ph_p1 - ph_m1)[..., None]
    with np.errstate(divide='ignore'):
        log_ph_p1 = np.log(ph_p1)[..., None]
        log_ph_m1 = np.log(ph_m1)[..., None]
    s2 = estimate_variance(x0, **kw)[..., None]

    def llr(u):
        """log q_hat / f at offset u = y - mu."""
        return np.logaddexp(
            np.logaddexp(log_ph0 + 0 * u, log_ph_p1 + (2 * u - 1) / (2 * s2)),
            log_ph_m1 + (-2 * u - 1) / (2 * s2),
        )

    t, w = np.polynomial.hermite.hermgauss(nodes)
    w = w / np.sqrt(np.pi)
    u = np.sqrt(2 * s2) * t  # nodes of the cover distribution

    # moments under cover H0
    l0 = llr(u)
    m0 = np.sum(w * l0, axis=-1)
    v0 = np.sum(w * l0**2, axis=-1) - m0**2
    # mean under stego H1: cover sample, then +1/-1 with the actual p_p1/p_m1
    m1 = (
        (1 - p_p1 - p_m1) * m0
        + p_p1 * np.sum(w * llr(u + 1), axis=-1)
        + p_m1 * np.sum(w * llr(u - 1), axis=-1)
    )
    return float((np.sum(m1) - np.sum(m0)) / np.sqrt(np.sum(v0)))
