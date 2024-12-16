
import numpy as np

from .. import tools


def estimate_beta(h0: np.ndarray, h1: np.ndarray) -> float:
    """Estimates beta from cover and stego histograms.

    Implements Eq. 3 from

    Fridrich, Goljan, Hogea:
    Steganalysis of JPEG images: Breaking the F5 Algorithm.
    Information Hiding, 2003.

    :param h0: cover histogram, estimated for instance using cartesian calibration
    :type h0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param h1: stego histogram
    :type h1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: change rate estimate
    :rtype: float

    :Example:

    >>> y1 = jpeg1.Y
    >>> y2 = sw.utils.calibration.cartesian(y1, jpeg1.qt)
    >>> h1, _ = np.histogram(np.abs(y1[..., 0, 1]).flatten(), 3, range=(0, 3))
    >>> h2, _ = np.histogram(np.abs(y2[..., 0, 1]).flatten(), 3, range=(0, 3))
    >>> beta_hat = sw.F5.estimate_beta(h1, h2)
    """
    beta_hat = (
        h0[1] * (h1[0] - h0[0]) +
        (h1[1] - h0[1]) * (h0[2] - h0[1])
    ) / (h0[1]**2 + (h0[2] - h0[1])**2)
    return beta_hat


def attack(
    y1: np.ndarray,
    qt: np.ndarray,
    **kw,
) -> float:
    """Runs a histogram attack with cartesian callibration, targetted against F5.

    Pools the estimates for the DCT AC modes 01, 10, and 11.

    :param y1: Stego DCT coefficients.
    :type y1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param qt: quantization table
    :type qt: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: change rate estimate
    :rtype: float

    :Example:

    >>> beta_hat = sw.F5.attack(jpeg1.Y, jpeg1.qt[0])
    """

    # cartesian calibration
    y2 = tools.calibration.cartesian(y1, qt=qt, **kw)

    # histograms
    h1_01, _ = np.histogram(np.abs(y1[:, :, 0, 1]).flatten(), 3, range=(0, 3))
    h1_10, _ = np.histogram(np.abs(y1[:, :, 1, 0]).flatten(), 3, range=(0, 3))
    h1_11, _ = np.histogram(np.abs(y1[:, :, 1, 1]).flatten(), 3, range=(0, 3))
    h2_01, _ = np.histogram(np.abs(y2[:, :, 0, 1]).flatten(), 3, range=(0, 3))
    h2_10, _ = np.histogram(np.abs(y2[:, :, 1, 0]).flatten(), 3, range=(0, 3))
    h2_11, _ = np.histogram(np.abs(y2[:, :, 1, 1]).flatten(), 3, range=(0, 3))

    # change rate estimates
    beta_01 = estimate_beta(h0=h2_01, h1=h1_01)
    beta_10 = estimate_beta(h0=h2_10, h1=h1_10)
    beta_11 = estimate_beta(h0=h2_11, h1=h1_11)

    # pool estimates
    beta_hat = np.mean([beta_01, beta_10, beta_11])
    beta_hat = np.clip(beta_hat, 0, None)
    return beta_hat

    # beta_01 = (h2_01[1] * (h1_01[0] - h2_01[0]) + (h1_01[1] - h2_01[1]) * (h2_01[2] - h2_01[1])) / (h2_01[1]**2 + (h2_01[2] - h2_01[1])**2)
    # beta_10 = (h2_10[1] * (h1_10[0] - h2_10[0]) + (h1_10[1] - h2_10[1]) * (h2_10[2] - h2_10[1])) / (h2_10[1]**2 + (h2_10[2] - h2_10[1])**2)
    # beta_11 = (h2_11[1] * (h1_11[0] - h2_11[0]) + (h1_11[1] - h2_11[1]) * (h2_11[2] - h2_11[1])) / (h2_11[1]**2 + (h2_11[2] - h2_11[1])**2)

    # #
    # h1_01, _ = np.histogram(y1[:, :, 0, 1].flatten(), 8, range=(-4, 4))
    # h1_10, _ = np.histogram(y1[:, :, 1, 0].flatten(), 8, range=(-4, 4))
    # h1_11, _ = np.histogram(y1[:, :, 1, 1].flatten(), 8, range=(-4, 4))
    # h2_01, _ = np.histogram(y2[:, :, 0, 1].flatten(), 8, range=(-4, 4))
    # h2_10, _ = np.histogram(y2[:, :, 1, 0].flatten(), 8, range=(-4, 4))
    # h2_11, _ = np.histogram(y2[:, :, 1, 1].flatten(), 8, range=(-4, 4))
    # #
    # beta_01 = (h1_01[0+4] - h2_01[0+4]) / (h2_01[-1+4] + h2_01[1+4])
    # beta_10 = (h1_10[0+4] - h2_10[0+4]) / (h2_10[-1+4] + h2_10[1+4])
    # beta_11 = (h1_11[0+4] - h2_11[0+4]) / (h2_11[-1+4] + h2_11[1+4])

