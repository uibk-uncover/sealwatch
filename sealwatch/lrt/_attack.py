"""Implements Gaussian LRT from MiPOD.

Authors: Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np
import scipy.signal

from ..utils import EPS


def local_variance(
	spatial: np.ndarray,
) -> np.ndarray:
	""""""
	# local mean
	kernel = np.ones((3, 3), dtype='float')
	R = scipy.signal.convolve2d(np.ones_like(spatial), kernel, mode='same', boundary='fill')
	Ex = scipy.signal.convolve2d(spatial, kernel, mode='same', boundary='fill') / R

	# local variance
	Ex2 = scipy.signal.convolve2d(spatial**2, kernel, mode='same', boundary='fill') / R
	Vx = Ex2 - Ex**2
	return Vx

def attack(
    spatial: np.ndarray,
	# beta: np.ndarray,
	# gamma: np.ndarray = None,
	beta: float = None
) -> np.ndarray:
	#
	# spatial = spatial / 255.
	sigma2 = local_variance(spatial / 255.)
	#
	# if gamma is None:
	# 	gamma = np.ones_like(spatial)
	# 	if beta is not None:  # knows rate
	# 		gamma = gamma * beta
	# 	else:  # rate 1/N
	# 		gamma = gamma / gamma.size
	#
	# defl = 1 - 2*gamma + gamma*(
	# 	np.exp((-spatial - .5/255)/(sigma2+1e-15)) +
	# 	np.exp((spatial - .5/255)/(sigma2+1e-15))
	# )
	# print(defl)
	# print(defl.shape)
	# return defl

	defl = np.sqrt(2 * np.sum(1/(sigma2**2+1e-15)*beta**2))
	# defl = np.sqrt(2) * np.sum(1/(sigma2**2+1e-15)*beta**2) / np.sqrt(np.sum(1/(sigma2**2+1e-15)) + 1e-10)
	return defl